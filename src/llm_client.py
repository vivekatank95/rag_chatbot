import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# HuggingFace endpoint with explicit parameters
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # public model
    task="text-generation",
    huggingfacehub_api_token=HF_API_KEY,
    temperature=0.1,
    max_new_tokens=512
)

# Wrap in ChatHuggingFace for chat interface
model = ChatHuggingFace(llm=llm_endpoint)

def generate_answer(prompt: str) -> str:
    """Generate answer using HuggingFace LLaMA via LangChain."""
    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        print("HuggingFace LLaMA API error:", e)
        return "Error generating response"


