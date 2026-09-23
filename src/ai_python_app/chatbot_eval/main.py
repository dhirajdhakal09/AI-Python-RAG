import os
from urllib import response

from dotenv import load_dotenv
from langsmith import Client
#from openai import OpenAI
import openai


def main() -> None:
    """Load environment variables and print the chatbot evaluation value."""
    load_dotenv()
    api_key = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    os.environ["OPENAI_KEY"]=os.getenv("OPENAI_KEY")
    print("LANGSMITH_API_KEY:", api_key)

    # create the dataset in LangSmith and evaluate the chatbot responses
    #chatbot_eval = ChatBotEval()
    #chatbot_eval.evaluate()

    openai_client = OpenAIClient()
    openai_client.correctness(
        inputs={"question": "What is LangChain?"},
        outputs={"response": "A framework for building LLM applications"},
        reference_outputs={"answer": "A framework for building LLM applications"}
    )
    ## call my_app for every data points 
    openai_client.evaluate()
    


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

#llm as a judge
from langsmith import wrappers
class OpenAIClient:
    eval_instructions = "You are an expert professor specialized in grading students answers to questions."
    def __init__(self):
        ##self.client = OpenAI(os.getenv("ANTHROPIC_API_KEY"))
        ##openai_key = os.getenv("OPENAI_KEY")
        self.anthropic_wrapper = wrappers.wrap_openai(openai.OpenAI())
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
        response=self.anthropic_wrapper.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                  {"role":"system","content":self.eval_instructions},
                  {"role":"user","content":user_content}
            ]
      ).choices[0].message.content

        return response == "CORRECT"


    """Run evaluate"""
    default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."
    def my_app(self,question: str, model: str = "gpt-4o-mini", instructions: str = default_instructions) -> str:
        return self.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
        ).choices[0].message.content


    
    ### Call my_app for every datapoints
    def ls_target(self, inputs: str) -> dict:
        return {"response": self.my_app(inputs["question"])}


    ## run evaluation
    def evaluate(self):
       experiment_results=self.langsmith_client.evaluate(
        self.ls_target, ## Your AI system
        data="Chatbot Evaluations 1",
        evaluators=[self.correctness,self.concision],
        experiment_prefix="openai-4o-mini-chatbot"
       )

       print("experiment_results:", experiment_results)

