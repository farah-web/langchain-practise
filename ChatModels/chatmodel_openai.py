from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model=ChatOpenAI(model='gpt-4o-mini')
res=model.invoke("What is the capital of France?")
print(res.content)