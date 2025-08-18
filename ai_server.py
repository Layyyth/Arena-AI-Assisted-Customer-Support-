# we'll load our model into the gpu and expose it throught api endpoint 
from typing import Optional
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import uvicorn
import json
import traceback

from prompts.prompt import BankingPrompts
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


def format_few_shot_prompt(user_input:str,customer_name:Optional[str],customer_id:Optional[str])-> str:
    ''' no data ? prompt engineering '''
    
    # system prompt
    prompt = BankingPrompts.SYSTEM_PROMPT
    
    # few shot example to guide the model
    prompt += "\n\n---Examples---"
    for example in BankingPrompts.EXAMPLES:
        example_prompt_text = BankingPrompts.get_ticket_generation_prompt(example["input"])
        prompt += f"\n\n{example_prompt_text}\nResponse:\n{example['output']}"
    
    # add current user query
    prompt+= "\n\n---Current request"
    final_user_prompt  = BankingPrompts.get_ticket_generation_prompt(user_input,customer_name,customer_id)
    prompt += f"\n\n{final_user_prompt}\nResponse:\n"
    
    return prompt


# -- api endpoint 
@app.post("/generate")
async def generate(request: Request):
    try:
        body = await request.json()
        user_input = body.get("user_input")
        customer_name = body.get("customer_name")
        customer_id = body.get("customer_id")

        if not user_input:
            return JSONResponse({"error": "user_input is required"}, status_code=400)

        print("Generating ticket with a single, comprehensive prompt..")
        final_prompt = format_few_shot_prompt(user_input, customer_name, customer_id)

        generation_params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=1024)
        generation_request_id = f"gen-{random_uuid()}"

        results_generator = engine.generate(final_prompt, generation_params, generation_request_id)
        generated_text = ""
        async for request_output in results_generator:
            if request_output.finished:
                generated_text = request_output.outputs[0].text
                break

        if not generated_text:
             return JSONResponse({"error":"Failed to generate output from model"},status_code=500)

        print(f"\n--- RAW MODEL OUTPUT ---\n{generated_text}\n------------------------\n")

        # Use the robust raw_decode method to parse the output
        start_index = generated_text.find('{')
        if start_index == -1:
            raise ValueError("No JSON object start '{' found in the model's output.")

        json_decoder = json.JSONDecoder()
        json_output, _ = json_decoder.raw_decode(generated_text[start_index:])
        return JSONResponse(content=json_output)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": f"An unexpected error occurred in the AI server: {e}"},
            status_code=500
        )
                    
    
if __name__  == "__main__":
    # run on port 8001 to avoid conflict with our main api on port 8000
    uvicorn.run(app,host='0.0.0.0',port=8001)


