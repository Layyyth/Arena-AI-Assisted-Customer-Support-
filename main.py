from fastapi import FastAPI, status, Request
from pydantic import BaseModel,Field
import pika
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
INCOMING_QUEUE = os.getenv("RABBITMQ_INCOMING_QUEUE", "ticket_requests")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Ticket System API",
    description="An API to submit user requests for AI processing.",
    version="1.0.0",
)

# --- Pydantic Model (no change) ---
class UserRequest(BaseModel):
    user_input: str
    customer_name: Optional[str] = Field(None, description="The customer's name, if available.")
    customer_id: Optional[str] = Field(None, description="The customer's ID, if available.")
    ticket_id: str = Field(..., description="The unique identifier for this ticket from the source system.")

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
        credentials = pika.PlainCredentials('user', 'password')
        parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        #  Ensure the queue exists.
        channel.queue_declare(queue=INCOMING_QUEUE, durable=True)

        #  Publish the message.
        message_body = user_request.model_dump_json()
        
        channel.basic_publish(
            exchange='',
            routing_key=INCOMING_QUEUE,
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
            
            
            