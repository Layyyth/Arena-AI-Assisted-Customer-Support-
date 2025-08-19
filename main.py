from fastapi import FastAPI, status, Request
from pydantic import BaseModel,Field
import pika
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")

EXCHANGE_NAME = os.getenv("RABBITMQ_EXCHANGE")
INCOMING_QUEUE = os.getenv("RABBITMQ_INCOMING_QUEUE")
INCOMING_ROUTING_KEY = os.getenv("RABBITMQ_INCOMING_ROUTING_KEY")


# --- Initialization ---
app = FastAPI(
    title="AI Ticket System API",
    description="An API to submit user requests for AI processing.",
    version="1.0.0",
)

# --- Pydantic Model 
class UserRequest(BaseModel):
    userInput: str
    customerName: Optional[str] = Field(None, description="The customer's name, if available.")
    customerId: Optional[str] = Field(None, description="The customer's ID, if available.")
    ticketId: str = Field(..., description="The unique identifier for this ticket from the source system.")

# --- API Endpoint (with self-contained connection logic) ---
@app.post("/api/v1/ticket", status_code=status.HTTP_202_ACCEPTED)
def create_ticket(user_request: UserRequest):
    """
    Accepts user input, establishes a connection to RabbitMQ,
    and publishes the message for processing.
    """
    connection = None
    try:
        # Establish a new connection for this specific request.
        credentials = pika.PlainCredentials(RABBITMQ_USER,RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            #virtual_host=RABBITMQ_VHOST,    ---> if arena specifies one add. 
            credentials=credentials
            )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)
        #  Ensure the queue exists.
        channel.queue_declare(queue=INCOMING_QUEUE, durable=True)
        
        channel.queue_bind(
            exchange=EXCHANGE_NAME, 
            queue=INCOMING_QUEUE, 
            routing_key=INCOMING_ROUTING_KEY
        )

        message_body = user_request.model_dump_json()

        channel.basic_publish(
            exchange=EXCHANGE_NAME,              # <-- Target the exchange
            routing_key=INCOMING_ROUTING_KEY,     # <-- Specify the "address"
            body=message_body.encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
            )
        )

        return {"status": "accepted", "message": "Your request has been queued for processing."}

    except pika.exceptions.AMQPError as e:
        return {"status": "error", "message": f"Failed to queue request: {e}"}, 500
    finally:
        if connection and not connection.is_closed:
            connection.close()