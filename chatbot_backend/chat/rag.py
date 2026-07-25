import os
from django.conf import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DB_DIR = "./chroma_db"

# Example knowledge base
# DOCUMENTS = [
#     "RAG stands for Retrieval Augmented Generation.",
#     "Chroma is a vector database for similarity search.",
#     "Ollama allows running LLMs locally.",
# ]
# Load PDF
pdf_path = os.path.join(settings.BASE_DIR, "main12.pdf")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Initialize splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Initialize or load persistent Chroma DB and add documents
vectorstore = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings
)

# Always try to add/update documents
if split_docs:
    vectorstore.add_documents(split_docs)
    vectorstore.persist()
    print("Documents added to Chroma DB.")
else:
    print("No documents to add to Chroma DB.")

