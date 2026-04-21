"""
AI Chatbot – OpenAI Version (v2)

This version:
- Uses the OpenAI API for real AI responses
- Logs every conversation to conversation_log.txt
- Keeps a simple, clean structure for learning
"""

import os
from openai import OpenAI

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def print_banner():
    print(
        "\n==============================\n"
        "      AI Chatbot (OpenAI)\n"
        "==============================\n"
        "Type 'quit' to exit.\n"
    )


def get_ai_response(user_input: str) -> str:
    """
    Calls OpenAI's API to generate a response.
    Falls back to a safe message if something goes wrong.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful learning-lab assistant."},
                {"role": "user", "content": user_input},
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"(AI error) Something went wrong: {e}"


def log_conversation(user_input: str, bot_response: str):
    with open("conversation_log.txt", "a") as log:
        log.write(f"You: {user_input}\n")
        log.write(f"Bot: {bot_response}\n")
        log.write("-" * 40 + "\n")


def main():
    print_banner()

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in ("quit", "exit"):
            print("Bot: Goodbye! See you next time.")
            break

        bot_response = get_ai_response(user_input)

        print(f"Bot: {bot_response}")

        log_conversation(user_input, bot_response)


if __name__ == "__main__":
    main()
