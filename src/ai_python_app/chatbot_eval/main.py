import os
from dotenv import load_dotenv
from langsmith import Client
from anthropic import Anthropic


def main() -> None:
    """Load environment variables and print the chatbot evaluation value."""
    load_dotenv()
    api_key = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_API_KEY"] = api_key or ""
    os.environ["LANGSMITH_TRACING"] = "true"

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Missing ANTHROPIC_API_KEY in environment.")

    # create the dataset in LangSmith and evaluate the chatbot responses
    openai_client = OpenAIClient()
    openai_client.evaluate()


if __name__ == "__main__":
    main()


class ChatBotEval:
    def __init__(self):
        self.api_key = os.getenv("LANGSMITH_API_KEY")
        self.tracing = os.getenv("LANGSMITH_TRACING")

    def evaluate(self):
        client = Client()

        dataset_name = "Chatbot Evaluations 1"
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


class OpenAIClient:
    eval_instructions = "You are an expert professor specialized in grading students answers to questions."

    def __init__(self):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in environment.")

        self.anthropic_client = Anthropic(api_key=anthropic_key)
        ##self.anthropic_wrapper = wrappers.wrap_anthropic(self.anthropic_client)
        self.langsmith_client = Client()

    def correctness(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        user_content = f"""You are grading the following question:
{inputs['question']}
Here is the real answer:
{reference_outputs['answer']}
You are grading the following predicted answer:
{outputs['response']}
Respond with CORRECT or INCORRECT:
Grade:
"""

        response = self.anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=50,
            #temperature=0,
            system=self.eval_instructions,
            messages=[{"role": "user", "content": user_content}],
        )

        content = response.content[0].text
        return content.strip().upper() == "CORRECT"

    default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."

    def my_app(self, question: str, model: str = "claude-3-5-haiku-20241022", instructions: str = default_instructions) -> str:
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=200,
            #temperature=0,
            system=instructions,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text

    def ls_target(self, inputs: str) -> dict:
        return {"response": self.my_app(inputs["question"])}

    def evaluate(self):
        experiment_results = self.langsmith_client.evaluate(
            self.ls_target,
            data="Chatbot Evaluations 1",
            evaluators=[self.correctness],
            experiment_prefix="anthropic-chatbot",
        )

        print("experiment_results:", experiment_results)

