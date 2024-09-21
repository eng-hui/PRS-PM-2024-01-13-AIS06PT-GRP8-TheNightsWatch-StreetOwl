import requests
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def call_gpt(messages, model="gpt-4o-mini", require_json=True):
    if require_json:
        response_format = {"type": "json_object" }
    else:
        response_format = None
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format
    )
    return completion.choices[0].message.content