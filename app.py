from fastapi import FastAPI

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import create_input
from src.classes.CustomLogitsProcessor import NoNumberLogitsProcessor
from src.classes.Logger import logger
import torch

app = FastAPI()
logger.info("APP_INITIALIZED")

# If CUDA available use bnb
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype="bfloat16"
# )

model_name = "SoberSinceToday/misis_final"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                             # quantization_config=bnb_config,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
logits_processor = [NoNumberLogitsProcessor(tokenizer)]

logger.info("MODEL_PREPARED")


@app.get("/generate")
def generate(prompt: str):
    logger.info("SERVER_GOT_PROMPT")
    output = create_input(text=prompt, model=model, tokenizer=tokenizer,
                          logits_processor=logits_processor)
    logger.info("SERVER_SENT_OUTPUT")
    return output
