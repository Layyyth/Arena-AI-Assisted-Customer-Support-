import pika
import time
import json
import redis
import requests

# -- initilaize redis connection 
# connect to the redis instance running in our docker container , db=0 selects default db
try:
    r=redis.Redis(host='localhost',port=6379,db=0,decode_responses=True)
    r.ping() #connection checker
    print("successfully connected to redis")
except redis.exceptions.ConnectionError as e:
    print(f"couldn't connect to redis {e}")
    exit(1) 



# -- rabbit connection -- 
# this loop will keep trying to connect to rabbitmq until it succeeds, this is useful
# because our worker might start b4 rabbitmq is fully ready in docker.

credentials = pika.PlainCredentials('user','password')
parameters = pika.ConnectionParameters('localhost','5672','/',credentials)

connection = None
while not connection:
    try:
        connection = pika.BlockingConnection(parameters)
        print("successfully connected to RabbitMQ")
    except pika.exceptions.AMQPConnectionError:
        print("connection to RabbitMQ failed. Retrying in 5 Seconds ..")
        time.sleep(5)

channel = connection.channel()
channel.queue_declare(queue="ticket_requests",durable=True)
print('Waiting for messages in queue "ticket_requests". to exit press CTRL+C ')


VLLM_API_URL = "http://localhost:8001/generate"

def get_ai_ticket(user_input:str)-> str:
    '''calls vllm server to generate the structured ticket, returns the generated json as a string or None if it fails'''
    headers = {"Content-Type":"application/json"}
    data = {"user_input":user_input}

    try:
        response = requests.post(VLLM_API_URL,headers=headers,json=data,timeout=30)
        response.raise_for_status() # raise error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        print(f" error : couldn't connect to the vLLM server :{e}")
        # try to requeue but for now
        return None

# -- processing locic -- 
def process_user_request(user_input):

    #we'll use user input as key of our cache 
    cache_key = f"ticket:{user_input}"
    cahced_results = r.get(cache_key)

    if cahced_results :
        print(f"cache hit for key : '{cache_key}'")
        # in real system we should return it to the arena backend
        print(json.loads(cahced_results))
        return
    
    print(f" Cache MISS for key: '{cache_key}'")
    print("Calling vLLM Inference Server...")
    
    
    result_json_str = get_ai_ticket(user_input)
    
    if result_json_str:
        try:
            # we validate is JSON here before printing or caching
            parsed_json = json.loads(result_json_str)
            print(f"vLLM processing complete")
            print(json.dumps(parsed_json,indent=2))
            
            r.set(cache_key,result_json_str,ex=3600)
            print(f"stored new result in cache")
            
        except json.JSONDecodeError:
            print(f" ERROR: AI Server returned a non-JSON response: {result_json_str}")
    else:
        print(" Failed to get a valid response from the AI server.")

    
# -- callback function 
# this function is called by pika whenever a message is received from the queue
def callback(ch,method,properties,body):
    # note : body is the message content whcich is in bytes we decode it to string
    user_input_str = body.decode()
    print(f"[x] Received user input: '{user_input_str}'")
    process_user_request(user_input_str)
    # to acknowledge the message this tells rabbitmq that we've successfully processed it and it can be safely deleted
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print("task acknowledged .. ")



# -- start consuming --
# this tells the channel to start listening on the 'ticket_requests' queue and calls our 'callback' function for each message
channel.basic_consume(queue='ticket_requests',on_message_callback=callback)
channel.start_consuming()


