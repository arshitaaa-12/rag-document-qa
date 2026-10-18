cat << 'EOF' > app.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = TextLoader("data/sample.txt")
docs = loader.load()
print(f"Loaded {len(docs)} document(s).")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = PromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context: {context}

Question: {question}
"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n🤖 Bot ready. Type 'quit' to exit.\n")
while True:
    q = input("You: ").strip()
    if q.lower() == "quit":
        break
    try:
        print(f"Bot: {rag_chain.invoke(q)}\n")
    except Exception as e:
        print(f"Error: {e}\n")
EOF