import os

from dotenv import load_dotenv
from langsmith import Client


def main() -> None:
    """Load environment variables and print the chatbot evaluation value."""
    load_dotenv()
    api_key = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    print("LANGSMITH_API_KEY:", api_key)
    chatbot_eval = ChatBotEval()
    chatbot_eval.evaluate()

if __name__ == "__main__":
    main()

class ChatBotEval:
    def __init__(self):
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.tracing = os.getenv("LANGSMITH_TRACING")

    def evaluate(self):
        client = Client()

        dataset_name ="Chatbot Evaluations 1"
        dataset = client.create_dataset(dataset_name)
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": "What is LangChain?"},
                    "outputs": {"answer": "A framework for building LLM applications"},
                },
                {
                    "inputs": {"question": "What is LangSmith?"},
                    "outputs": {"answer": "A platform for observing and evaluating LLM applications"},
                },
                {
                    "inputs": {"question": "What is OpenAI?"},
                    "outputs": {"answer": "A company that creates Large Language Models"},
                },
                {
                    "inputs": {"question": "What is Google?"},
                    "outputs": {"answer": "A technology company known for search"},
                },
                {
                    "inputs": {"question": "What is Mistral?"},
                    "outputs": {"answer": "A company that creates Large Language Models"},
                }
            ]
        )