from typing import Optional
import pika
import time
import json
import redis
import requests
import os
from dotenv import load_dotenv


class TicketWorker:
    
    '''class presenting a worker that consumes ticket requests from RabbitMQ, process them using AI model, and caches the results on reids'''
    
    def __init__(self):
        self.redis_host = os.getenv("REDIS_HOST","localhost")
        self.rabbit_host = os.getenv("RABBITMQ_HOST","localhost")
        self.rabbit_user = os.getenv("RABBITMQ_USER","user")
        self.rabbit_pass = os.getenv("RABBITMQ_PASS","password")
        self.vllm_api_url = os.getenv("VLLM_API_URL","http://localhost:8001/generate")
        
        self.incoming_queue = os.getenv("RABBITMQ_INCOMING_QUEUE", "ticket_requests")
        self.outgoing_queue = os.getenv("RABBITMQ_OUTGOING_QUEUE", "processed_tickets")
        
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
        self.rabbit_channel.queue_declare(queue=self.incoming_queue, durable=True)
        self.rabbit_channel.queue_declare(queue=self.outgoing_queue, durable=True)
        
        
    def _publish_processed_ticket(self,processed_ticket:dict):
        try:
            message_body=json.dumps(processed_ticket)
            self.rabbit_channel.basic_publish(
                exchange='',
                routing_key=self.outgoing_queue,
                body=message_body.encode('utf-8'),
                properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
            )
            ticket_id = processed_ticket.get('ticket_id','[N/A]')
            print(f"publish process ticket {ticket_id} to '{self.outgoing_queue}' qeueue.")
        except Exception as e:
            print(f"!! Failed to publish process ticket to RabbitMQ : {e}")
        
        
        
    def _get_ai_ticket(self, user_input: str, customer_name: Optional[str], customer_id: Optional[str]) -> str:
        '''calls vllm server to generate the  structured ticket'''
        headers = {"Content-Type":"application/json"}
        data = {
            "user_input":user_input,
            "customer_name": customer_name,
            "customer_id": customer_id
        }
        try:
            response = requests.post(self.vllm_api_url,headers=headers,json=data,timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"ERROR : Cloud not get response from vLLM : {e}")
            return None
    
    def process_user_request(self, user_input: str, customer_name: Optional[str], customer_id: Optional[str], ticket_id: Optional[str]):
        '''main processing logic including caching'''
        cache_key = f"ticket:{user_input}"
        cached_results = self.redis_conn.get(cache_key)
        
        if cached_results :
            print(f"Cache hit for key : '{cache_key}'")
            processed_ticket = json.loads(cached_results)
            processed_ticket["ticket_id"] = ticket_id
            print(json.dumps(json.loads(cached_results),indent=2))
            self._publish_processed_ticket(processed_ticket)
            return
        
        print(f"cache miss for key :'{cache_key}'")
        print("Calling vLLM inference Server ..")        
        
        result_json_str = self._get_ai_ticket(user_input, customer_name, customer_id)
        
        if result_json_str:
            try:
                parsed_json = json.loads(result_json_str)
                
                if ticket_id:
                    parsed_json["ticket_id"] = ticket_id
                    
                print("vLLM processing complete!")
                print(json.dumps(parsed_json, indent=2))

                self._publish_processed_ticket(parsed_json)
                
        
                if "error" not in parsed_json:
                    ai_part_to_cache = parsed_json.copy()
                    ai_part_to_cache.pop("ticket_id", None)
                    self.redis_conn.set(cache_key, json.dumps(ai_part_to_cache), ex=3600)
                    print("Stored new AI result in cache.")

            except json.JSONDecodeError:
                print(f"ERROR: AI server returned a non-JSON response: {result_json_str}")
        else:
            print("Failed to get a valid response from AI server.")
            
    
    def callback(self,ch,method,properties,body):
        
        try:
            message_data = json.loads(body.decode())
            user_input = message_data.get("user_input")
            customer_name = message_data.get("customer_name")
            customer_id = message_data.get("customer_id")
            ticket_id = message_data.get("ticket_id")
            
            if user_input and ticket_id:
                print(f"\n Received ticket {ticket_id} for input: '{user_input}'")
                self.process_user_request(user_input, customer_name, customer_id, ticket_id)
            else:
                print("Received message without 'user_input' or 'ticket_id'. Discarding.")
        
        except json.JSONDecodeError:
            print(f"Received invalid JSON message. Discarding: {body.decode()}")
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Task Acknowledged!")
        
        
    def run(self):
        '''this start woker and begins consumung messages'''
        self._connect_rabbitmq()
        self._connect_redis()
        
        print('waiting for messages in queue "ticket_requests". To exit press CTRL+C')
        self.rabbit_channel.basic_consume(queue="ticket_requests",on_message_callback=self.callback)
        
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


