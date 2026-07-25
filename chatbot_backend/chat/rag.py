import logging
import os
from django.conf import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR = os.path.join(settings.BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_base"


def initialize_vectorstore():
    try:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        logger.exception("Failed to initialize embedding model")
        raise RuntimeError(f"Embedding initialization failed: {exc}") from exc

    pdf_path = os.path.join(settings.BASE_DIR, "main12.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as exc:
        logger.exception("Failed to load PDF")
        raise RuntimeError(f"PDF loading failed: {exc}") from exc

    if not documents:
        logger.warning("No pages were found in the PDF")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)
    except Exception as exc:
        logger.exception("Failed to split PDF into chunks")
        raise RuntimeError(f"Document splitting failed: {exc}") from exc

    for index, doc in enumerate(split_docs):
        doc.metadata["source"] = os.path.basename(pdf_path)
        doc.metadata["chunk_id"] = index
        doc.metadata["chunk_size"] = 500

    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    except Exception as exc:
        logger.exception("Failed to initialize Chroma vector store")
        raise RuntimeError(f"Chroma initialization failed: {exc}") from exc

    if split_docs:
        try:
            vectorstore.add_documents(split_docs)
            vectorstore.persist()
            logger.info("Saved %s document chunks to Chroma DB at %s", len(split_docs), CHROMA_DB_DIR)
        except Exception as exc:
            logger.exception("Failed to save chunks to Chroma")
            raise RuntimeError(f"Chroma save failed: {exc}") from exc
    else:
        logger.warning("No documents to add to Chroma DB.")

    return vectorstore


try:
    initialize_vectorstore()
except Exception as exc:
    logger.error("RAG setup failed: %s", exc)

