from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# -----------------------------
# 1. Load YouTube transcript
# -----------------------------
video_id = "Gfr50f6ZBvo"

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(item.text for item in transcript_list)

except TranscriptsDisabled:
    raise RuntimeError("Transcripts are disabled for this video")

except NoTranscriptFound:
    raise RuntimeError("No transcript found for this video")

# -----------------------------
# 2. Split transcript
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.create_documents([transcript])

# -----------------------------
# 3. Embeddings + Vector Store
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -----------------------------
# 4. Prompt
# -----------------------------
prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.
Answer ONLY from the context below.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# -----------------------------
# 5. RAG Pipeline (Runnable-based)
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
    )
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    | StrOutputParser()
)

# -----------------------------
# 6. Invoke
# -----------------------------
question = "Can you summarize the video?"
result = rag_chain.invoke(question)

print("Answer:\n", result)
