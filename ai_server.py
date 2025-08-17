# we'll load our model into the gpu and expose it throught api endpoint 
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import uvicorn
import json
import traceback


#-- define engine args
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"

engine_args = AsyncEngineArgs(
    model = MODEL_NAME,
    quantization = "gptq",
    gpu_memory_utilization=0.60,
    max_model_len=2048
)

#-- create vllm engine and fastapi app
engine = AsyncLLMEngine.from_engine_args(engine_args)
app = FastAPI(title="vLLM inference Server for Banking")


#-- prompt time (go ham)
#-------------------------------------------------------------------------------------------------------
PROMPT_TEMPLATE = """
A user has submitted a support request for a financial institution. Your task is to analyze the request and convert it into a structured JSON object.

The JSON output must have two main keys: "ticket" and "solution".

The "ticket" object must contain exactly these keys:
- "ticket_type": Must be one of "complaint", "inquiry", or "assistance".
- "title": A brief, descriptive summary of the user's issue.
- "description" : a more detailed description of the input explaining the issue.
- "severity": Must be one of "low", "medium", "high", or "critical".
- "department_impacted": The most relevant bank department (e.g., "Digital Banking", "Card Services", "Loans", "Customer Accounts").
- "service_impacted": The specific banking service affected (e.g., "Mobile Banking", "Credit Card", "Online Banking", "Mortgage Application").

The "solution" object must contain exactly these keys:
- "solution_type": A high-level category for the solution (e.g., "Technical Support & App Issue Resolution", "Account Information Inquiry").
- "solution_steps": An array of strings outlining the initial actions to take.

User Request:
"{user_input}"

Strictly provide only the JSON object as the output.
JSON Output:
"""



# -- api endpoint 
@app.post("/generate")
async def generate(request : Request):
    body = await request.json()
    user_input = body.get("user_input")

    if not user_input:
        return JSONResponse({"error": "user_input is required"},status_code=400)
    
    final_prompt = PROMPT_TEMPLATE.format(user_input =user_input)
    
    sampling_params = SamplingParams(
        temperature=0.4,
        top_p= 0.95,
        max_tokens=1024
    )

    request_id = f"cmpl-{random_uuid()}"
    result_generator = engine.generate(final_prompt,sampling_params,request_id)

    final_output = None
    async for request_output in result_generator:
        if request_output.finished:
            final_output = request_output
            break

    if final_output is None:
        return JSONResponse({"error":"Failed to generate output from model"},status_code=500)
    
    generated_text = final_output.outputs[0].text

    # parse model output as JSON
    try:
        start_index = generated_text.find('{')        
        if start_index == -1 :
            raise ValueError("No valid JSON object found in the model's output.")
            
        json_decoder = json.JSONDecoder()
        json_output, _ = json_decoder.raw_decode(generated_text[start_index:])
        return JSONResponse(content=json_output)
    except Exception as e:
        print("\n -- detailed traceback --")
        traceback.print_exc()
        print("\n ---------------------------")
        return JSONResponse(
            {"error": f"Model generated invalid or incomplete JSON: {e}", "raw_output": generated_text},
            status_code=500
        )
    
if __name__  == "__main__":
    # run on port 8001 to avoid conflict with our main api on port 8000
    uvicorn.run(app,host='0.0.0.0',port=8001)


