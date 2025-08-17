from fastapi import FastAPI, status, Request
from pydantic import BaseModel
import pika

# We no longer need the lifespan manager for RabbitMQ,
# so the startup is much simpler.

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Ticket System API",
    description="An API to submit user requests for AI processing.",
    version="1.0.0",
)

# --- Pydantic Model (no change) ---
class UserRequest(BaseModel):
    user_input: str

# --- API Endpoint (with self-contained connection logic) ---
@app.post("/api/v1/ticket", status_code=status.HTTP_202_ACCEPTED)
def create_ticket(user_request: UserRequest):
    """
    Accepts user input, establishes a connection to RabbitMQ,
    and publishes the message for processing.
    """
    connection = None
    try:
        # --- THIS IS THE NEW, SAFE CONNECTION LOGIC ---
        # 1. Establish a new connection for this specific request.
        credentials = pika.PlainCredentials('user', 'password')
        parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # 2. Ensure the queue exists.
        channel.queue_declare(queue='ticket_requests', durable=True)

        # 3. Publish the message.
        message_body = user_request.user_input
        channel.basic_publish(
            exchange='',
            routing_key='ticket_requests',
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