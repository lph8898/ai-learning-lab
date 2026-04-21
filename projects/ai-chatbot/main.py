"""
Simple Console Chatbot (Starter Version)

This chatbot:
- Runs in the terminal
- Uses simple rules to respond
- Is designed to be upgraded later with real AI APIs
"""

import textwrap


def print_banner():
    banner = """
    ==============================
          AI Chatbot (Demo)
    ==============================
    Type 'quit' to exit.
    """
    print(textwrap.dedent(banner))


def get_bot_response(user_input: str) -> str:
    """
    Very simple rule-based responses.
    Later, you can replace this with a real AI model call.
    """
    text = user_input.lower().strip()

    if text in ("hi", "hello", "hey"):
        return "Hello! I'm your learning-lab chatbot. What are you exploring today?"
    if "help" in text:
        return "You can ask me about AI learning, projects, or just chat casually."
    if "bbc" in text:
        return "Brisbane Boys’ College is a great school with strong academics."
    if "ai" in text and "learn" in text:
        return "A good way to learn AI is to build small projects, just like this one."
    if "thank" in text:
        return "You're welcome! Keep experimenting—that's how you get good at this."
    if text in ("quit", "exit"):
        return "quit"  # special signal to exit

    # Default fallback
    return "Interesting! Tell me more, or ask me something about AI or projects."


def main():
    print_banner()

    while True:
        user_input = input("You: ")

        if not user_input.strip():
            continue

        response = get_bot_response(user_input)

        if response == "quit":
            print("Bot: Goodbye! See you back in the AI Learning Lab.")
            break

        print(f"Bot: {response}")

        # Save conversation to a log file
        with open("conversation_log.txt", "a") as log:
            log.write(f"You: {user_input}\n")
            log.write(f"Bot: {response}\n")
            log.write("-" * 40 + "\n")


if __name__ == "__main__":
    main()
