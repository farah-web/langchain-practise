from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=50
)

llm = HuggingFacePipeline(pipeline=pipe)
res=llm.invoke("What is the capital of Germany?")
print(res)
