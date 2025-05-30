import gradio as gr
import requests
from utils import create_prompt
from src.classes.Logger import logger

url = "http://127.0.0.1:8000/generate"
retrieved = []


def chat_with_model(message):
    prompt = create_prompt(data={"instruction": message, "retrieved_context": retrieved, "output": ""})['text']
    params = {'prompt': prompt}

    logger.info(f"CLIENT_GOT_PROMPT: {prompt}")

    response = requests.get(url, params=params).json()
    response = response[response.rfind("Пользователь2") + len("Пользователь2: "):]

    logger.info(f"CLIENT_GOT_OUTPUT: {response}")

    retrieved.append({'instruction': message, 'output': response})
    return response


iface = gr.Interface(fn=chat_with_model,
                     inputs=["text"],
                     outputs=["text"],
                     title="Example")
logger.info("GUI STARTED")
iface.launch()
