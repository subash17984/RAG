from rest_framework.decorators import api_view
from rest_framework.response import Response
from .rag import vectorstore
from .llm import llm

@api_view(["POST"])
def chat_view(request):
    query = request.data.get("message")

    # Use vectorstore directly (public API)
    docs = vectorstore.similarity_search(query, k=4)

    # Combine retrieved chunks
    context = "\n".join([d.page_content for d in docs])

    # Build prompt for Ollama
    prompt = f"""
    Use the context below to answer the question. Context:{context} Question:{query}"""

    # Generate answer
    answer = llm.invoke(prompt)

    return Response({"answer": answer})
