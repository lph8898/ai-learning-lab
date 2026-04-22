#!/usr/bin/env python3
"""
ai-learning-lab/projects/ai-chatbot/main.py

Simple local LLM chatbot using Hugging Face transformers (free models).
Optional: set HF_API_TOKEN to use Hugging Face Inference API instead of local model.
"""

import os
import sys
import json
import time
import logging
from typing import Optional

# Try to import transformers; if not installed, the user will install via requirements.txt
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

import requests

# Configuration
MODEL_NAME = os.environ.get("LOCAL_MODEL", "google/flan-t5-small")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # optional
LOG_FILE = os.environ.get("CHAT_LOG", "conversations.log")
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "512"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)


class SimpleChatbot:
    def __init__(self, model_name: str = MODEL_NAME, hf_token: Optional[str] = HF_API_TOKEN):
        self.model_name = model_name
        self.hf_token = hf_token
        self.generator = None
        if hf_token:
            logging.info("HF_API_TOKEN detected: will use Hugging Face Inference API for generation.")
        else:
            if pipeline is None:
                raise RuntimeError("transformers not installed. See requirements.txt and install dependencies.")
            logging.info(f"Loading local model: {model_name} (this may take a moment)")
            # Use a seq2seq/text2text pipeline for instruction models like FLAN-T5
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
                logging.info("Local model loaded successfully.")
            except Exception as e:
                logging.error("Failed to load local model: %s", e)
                logging.info("Falling back to Hugging Face Inference API if HF_API_TOKEN is set.")
                self.generator = None

    def generate_local(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using local transformers pipeline."""
        if not self.generator:
            raise RuntimeError("Local generator not available.")
        # Keep prompt length reasonable
        prompt = prompt.strip()
        if len(prompt) > MAX_INPUT_LENGTH:
            prompt = prompt[-MAX_INPUT_LENGTH:]
        outputs = self.generator(prompt, max_length=max_length, do_sample=False)
        if isinstance(outputs, list) and outputs:
            return outputs[0].get("generated_text", "").strip()
        return str(outputs).strip()

    def generate_hf_api(self, prompt: str) -> str:
        """Generate text using Hugging Face Inference API (requires HF_API_TOKEN)."""
        if not self.hf_token:
            raise RuntimeError("HF_API_TOKEN not set.")
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # API returns list of generated outputs for text-generation/text2text
        if isinstance(data, list) and data:
            # Hugging Face returns [{'generated_text': '...'}]
            return data[0].get("generated_text", "").strip()
        # If the response is a dict with 'error', raise
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF API error: {data['error']}")
        return str(data).strip()

    def respond(self, prompt: str) -> str:
        """Unified respond method: prefer local model, fallback to HF API if available."""
        logging.info("User prompt: %s", prompt)
        try:
            if self.generator:
                reply = self.generate_local(prompt)
            elif self.hf_token:
                reply = self.generate_hf_api(prompt)
            else:
                raise RuntimeError("No generation method available. Install transformers or set HF_API_TOKEN.")
        except Exception as e:
            logging.exception("Generation failed: %s", e)
            reply = "Sorry, I couldn't generate a response right now."
        self.log_conversation(prompt, reply)
        return reply

    def log_conversation(self, user_text: str, bot_text: str):
        entry = {"timestamp": time.time(), "user": user_text, "bot": bot_text}
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logging.exception("Failed to write conversation log.")

# Simple CLI interface
def main():
    print("ai-learning-lab chatbot (local model). Type 'exit' to quit.")
    bot = SimpleChatbot()
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        response = bot.respond(user_input)
        print("\nBot:", response)


if __name__ == "__main__":
    main()
