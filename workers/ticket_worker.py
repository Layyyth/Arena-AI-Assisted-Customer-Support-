from typing import Optional
import pika
import time
import json
import redis
import requests
import os
from dotenv import load_dotenv



load_dotenv()

class TicketWorker:
    
    '''class presenting a worker that consumes ticket requests from RabbitMQ, process them using AI model, and caches the results on reids'''
    
    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST","localhost")
        self.rabbit_host = os.getenv("RABBITMQ_HOST","localhost")
        self.rabbit_user = os.getenv("RABBITMQ_USER","user")
        self.rabbit_pass = os.getenv("RABBITMQ_PASS","password")
        self.vllm_api_url = os.getenv("VLLM_API_URL","http://localhost:8001/generate")
        
        print(f"--- Worker configured to use AI Server at: {self.vllm_api_url} ---")
        
        self.exchange_name = os.getenv("RABBITMQ_EXCHANGE", "arena.assistance.exchange")
        self.incoming_queue = os.getenv("RABBITMQ_INCOMING_QUEUE", "arena.assistance.classification.queue")
        self.incoming_routing_key = os.getenv("RABBITMQ_INCOMING_ROUTING_KEY", "assistance.classify")
        self.outgoing_queue = os.getenv("RABBITMQ_OUTGOING_QUEUE", "arena.assistance.reason.queue")
        self.outgoing_routing_key = os.getenv("RABBITMQ_OUTGOING_ROUTING_KEY", "assistance.reason")
        
        self.redis_conn = None
        self.rabbit_conn = None
        self.rabbit_channel = None
        
    
    def _connect_redis(self):
        print("Attempting to Connect Redis .. ")
        try:
            self.redis_conn  = redis.Redis(host=self.redis_host,port=6379,db=0,decode_responses=True)
            self.redis_conn.ping()
            print("successfully connected to redis")
        except redis.exceptions.ConnectionError as e:
            print(f"couldn't connect to Redis:{e}")
            exit(1)
            
    
    def _connect_rabbitmq(self):
        print("Attempting to conenct to RabbitMQ.. ")
        credentials = pika.PlainCredentials(self.rabbit_user,self.rabbit_pass)
        paramaters = pika.ConnectionParameters(self.rabbit_host,5672,'/',credentials)
        
        while not self.rabbit_conn :
            try:
                self.rabbit_conn = pika.BlockingConnection(paramaters)
                print("successfully connected to RabbitMQ")
            except pika.exceptions.AMQPConnectionError :
                print("RabbitMQ connection failed , Retrying in 5 seconds ..")
                time.sleep(5)
                
        self.rabbit_channel = self.rabbit_conn.channel()
        self.rabbit_channel.exchange_declare(exchange=self.exchange_name, exchange_type='direct', durable=True)

        self.rabbit_channel.queue_declare(queue=self.incoming_queue, durable=True)
        self.rabbit_channel.queue_bind(exchange=self.exchange_name, queue=self.incoming_queue, routing_key=self.incoming_routing_key)

        self.rabbit_channel.queue_declare(queue=self.outgoing_queue, durable=True)
        self.rabbit_channel.queue_bind(exchange=self.exchange_name, queue=self.outgoing_queue, routing_key=self.outgoing_routing_key)
        
        
    def _publish_processed_ticket(self,processed_ticket:dict):
        try:
            message_body=json.dumps(processed_ticket)
            self.rabbit_channel.basic_publish(
                exchange=self.exchange_name,
                routing_key=self.outgoing_routing_key,
                body=message_body.encode('utf-8'),
                properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
            )
            ticketId = processed_ticket.get('ticketId','[N/A]')
            print(f"publish process ticket {ticketId} to exchange '{self.exchange_name}' with key '{self.outgoing_routing_key}'.")
        except Exception as e:
            print(f"!! Failed to publish process ticket to RabbitMQ : {e}")
        
        
        
    def _get_ai_ticket(self, userInput: str, customerName: Optional[str], customerId: Optional[str]) -> str:
        '''calls vllm server to generate the  structured ticket'''
        headers = {"Content-Type":"application/json"}
        data = {
            "userInput": userInput,
            "customerName": customerName,
            "customerId": customerId
        }
        try:
            response = requests.post(self.vllm_api_url,headers=headers,json=data,timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"ERROR : Cloud not get response from vLLM : {e}")
            return None
    
    def process_user_request(self, userInput: str, customerName: Optional[str], customerId: Optional[str], ticketId: Optional[str]):
        '''main processing logic including caching'''
        cache_key = f"ticket:{userInput}"
        cached_results = self.redis_conn.get(cache_key)
        
        if cached_results :
            print(f"Cache hit for key : '{cache_key}'")
            processed_ticket = json.loads(cached_results)
            processed_ticket["ticketId"] = ticketId
            print(json.dumps(json.loads(cached_results),indent=2))
            self._publish_processed_ticket(processed_ticket)
            return
        
        print(f"cache miss for key :'{cache_key}'")
        print("Calling vLLM inference Server ..")        
        
        result_json_str = self._get_ai_ticket(userInput, customerName, customerId)
        
        if result_json_str:
            try:
                parsed_json = json.loads(result_json_str)
                
                if ticketId:
                    parsed_json["ticketId"] = ticketId
                    
                print("vLLM processing complete!")
                print(json.dumps(parsed_json, indent=2))

                self._publish_processed_ticket(parsed_json)
                
        
                if "error" not in parsed_json:
                    ai_part_to_cache = parsed_json.copy()
                    ai_part_to_cache.pop("ticketId", None)
                    self.redis_conn.set(cache_key, json.dumps(ai_part_to_cache), ex=3600)
                    print("Stored new AI result in cache.")

            except json.JSONDecodeError:
                print(f"ERROR: AI server returned a non-JSON response: {result_json_str}")
        else:
            print("Failed to get a valid response from AI server.")
            
    
    def callback(self,ch,method,properties,body):
        
        try:
            message_data = json.loads(body.decode())
            userInput = message_data.get("userInput")
            customerName = message_data.get("customerName")
            customerId = message_data.get("customerId")
            ticketId = message_data.get("ticketId")
            
            if userInput and ticketId:
                print(f"\n Received ticket {ticketId} for input: '{userInput}'")
                self.process_user_request(userInput, customerName, customerId, ticketId)
            else:
                print("Received message without 'userInput' or 'ticketId'. Discarding.")
        
        except json.JSONDecodeError:
            print(f"Received invalid JSON message. Discarding: {body.decode()}")
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Task Acknowledged!")
        
        
    def run(self):
        '''this start woker and begins consumung messages'''
        self._connect_rabbitmq()
        self._connect_redis()
        
        print(f"waiting for messages in queue \"{self.incoming_queue}\". To exit press CTRL+C")
        self.rabbit_channel.basic_consume(queue=self.incoming_queue, on_message_callback=self.callback)
        
        try:
            self.rabbit_channel.start_consuming()
        except KeyboardInterrupt:
            if self.rabbit_conn and self.rabbit_conn.is_open:
                self.rabbit_channel.close()
            print("RabbitMQ connection Closed. Exiting")
            
if __name__ == "__main__":
    try:
        worker = TicketWorker()
        worker.run()        
    except Exception as e :
        print(f"an unexpected error occured : {e}")