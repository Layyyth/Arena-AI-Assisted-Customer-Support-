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
import logging

from prompts.prompt import BankingPrompts

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
app = FastAPI(title="vLLM inference Server for Banking")
engine = None
logger = logging.getLogger("vllm_server")


def format_few_shot_prompt(userInput:str,id:str)-> str:
    ''' no data ? prompt engineering '''
    
    # system prompt
    prompt = BankingPrompts.SYSTEM_PROMPT
    
    # few shot example to guide the model
    prompt += "\n\n---Examples---"
    for example in BankingPrompts.EXAMPLES:
        # We provide a placeholder ID for the examples since they are static
        example_prompt_text = BankingPrompts.get_ticket_generation_prompt(example["input"], "Example-ID-123")
        prompt += f"\n\n{example_prompt_text}\nResponse:\n{example['output']}"
    
    # add current user query
    prompt+= "\n\n---Current request"
    final_user_prompt  = BankingPrompts.get_ticket_generation_prompt(userInput,id)
    prompt += f"\n\n{final_user_prompt}\nResponse:\n"
    
    return prompt


@app.post("/generate")
async def generate(request: Request):
    id=None
    generation_request_id = None
    try:
        body = await request.json()
        userInput = body.get("userInput")
        id=body.get("id")

        
        if not userInput or not id:
            return JSONResponse({"error": "userInput and id are required"}, status_code=400)

        # generate unique id for the engine 
        generation_request_id = f"gen-{random_uuid()}"
        
        logger.info(f"Processing request for id: '{id}'. Assigned Generation id: '{generation_request_id}'")

        final_prompt = format_few_shot_prompt(userInput,id)
        generation_params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=1024)

        results_generator = engine.generate(final_prompt, generation_params, generation_request_id)
        generated_text = ""
        async for request_output in results_generator:
            if request_output.finished:
                generated_text = request_output.outputs[0].text
                break

        if not generated_text:
            logger.error(f"Failed to generate output for id: '{id}' (Generation id: '{generation_request_id}')") 
            return JSONResponse({"error":"Failed to generate output from model"},status_code=500)
        
        logger.info(f"Successfully generated response for id: '{id}' (Generation id: '{generation_request_id}')")
        logger.debug(f"Raw model output for id '{id}':\n{generated_text}")

        start_index = generated_text.find('{')
        if start_index == -1:
            raise ValueError("No JSON object start '{' found in the model's output.")

        json_decoder = json.JSONDecoder()
        json_output, _ = json_decoder.raw_decode(generated_text[start_index:])
        
        
        if "id" in json_output and json_output["id"] != id:
            logger.warning(f"Model hallucinated a different id for request id '{id}'. Overwriting with correct id.")
            json_output["id"] = id
        elif "id" not in json_output and "error" not in json_output:
             logger.warning(f"Model did not include id for request id '{id}'. Injecting correct id.")
             json_output["id"] = id
             
        return JSONResponse(content=json_output)

    except Exception as e:
        logger.error(f"Error processing id: '{id}' (Generation id: '{generation_request_id}'). Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            {"error": f"An unexpected error occurred in the AI server: {e}"},
            status_code=500
        )
                    
    
if __name__ == "__main__":
    
    logging.basicConfig(
        level =logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger.info("Initializing vLLM engine...")
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        quantization="gptq",
        gpu_memory_utilization=0.90,
        max_model_len=4096
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logger.info("Starting Uvicorn server on 0.0.0.0:8001...")
    uvicorn.run(app, host='0.0.0.0', port=8001)