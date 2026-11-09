

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langsmith import traceable,Client

import os

# load_dotenv()
load_dotenv(override=True)

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "QABot"

class QAResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question.")
    confidence:str = Field(..., description="The confidence level of the answer.")
    reasoning:str = Field(..., description="The reasoning behind the answer.")
    follow_up_questions:List[str]=Field(
        description="A List of follow-up questions related to the topic",
        default_factory=list
    )
    sources_needed:bool=Field(
        description="Indicates whether sources are needed for the answer",
        default=False
    )



class SmartQABot:
    def __init__(self,model_name:str="gemini-3.6-flash",temperature:float=0.3):
        self.model=ChatGoogleGenerativeAI(
            model=model_name,temperature=temperature,
            api_key=os.getenv("GEMINI_API_KEY")).with_structured_output(QAResponse)
        self.prompt=ChatPromptTemplate.from_messages([
            ("system","""You are a helpful assistant. answer the question in a structured format
            Your guidelines:
            -Answer questions accurately and concisely.
            -Be honest about uncertainity -set confidence to 'low' if unsure.
            -Provide reasoning for your answer.
            -If the answer is not known, respond with 'I don't know' and set confidence to 'low'.
            -If the answer requires sources, set sources_needed to True and provide a
        
            """),
            ("user","{question}")
        ])
        self.chain=self.prompt|self.model


    @traceable(name="ask_question",run_type="chain")
    def ask(self,question:str)->QAResponse:
        try:
            response=self.chain.invoke({"question":question})
            return response
        except Exception as e:
            return QAResponse(
                    answer="I don't know",
                    confidence="low",
                    reasoning=str(e),
                    follow_up_questions=["Could you please try again later"],
                    sources_needed=False
                )
            
    @traceable(name="ask_batch",run_type="chain")
    def ask_batch(self,questions:List[str])->List[QAResponse]:
        inputs=[{"question":q} for q in questions]
        try:
            responses=self.chain.batch(inputs)
            return responses
        except Exception as e:
            return [QAResponse(
                    answer="I don't know",
                    confidence="low",
                    reasoning=str(e),
                    follow_up_questions=["Could you please try again later"],
                    sources_needed=False
                ) for _ in questions]

questions=[
    "What is LangChain?",
    "What is the capital of France?",
    "Who won the 2020 US presidential election?"
]
def demo_qa_bot():
    bot=SmartQABot()
    question="What is LangChain?"
    for question in questions:
        response=bot.ask(question)
    response=bot.ask(question)
    print(f"Response for question '{question}': {response}")
    print("Question:",question)
    print("*"*50)
    print("Answer:",response.answer)
    print("Confidence:",response.confidence)
    print("Reasoning:",response.reasoning)
    print("Follow-up questions:",response.follow_up_questions)
    print("=="*50)


    questions=["What is LangChain?","What is the capital of France?","Who won the 2020 US presidential election?"]
    responses=bot.ask_batch(questions)
    print("\nBatch Responses:")
    for q,r in zip(questions,responses):
        print(f"Question '{q}':")
        print("Answer:",r.answer)
        print("Confidence:",r.confidence)
        print("Reasoning:",r.reasoning)
        print("Follow-up questions:",r.follow_up_questions)
        print("*"*50)

@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing():
    bot = SmartQABot()

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    responses = bot.ask_batch(questions)

    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f"  -> {r.answer[:100]}...")
        print(f"  Confidence: {r.confidence}")

    return responses

if __name__=="__main__":
    demo_qa_bot()
    demo_batch_processing()