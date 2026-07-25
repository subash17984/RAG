from rest_framework.decorators import api_view
from rest_framework.response import Response
from .rag import vectorstore
from .llm import llm

@api_view(["POST"])
def chat_view(request):
    try:
        query = request.data.get("message")

        if not query or not str(query).strip():
            return Response({"error": "Message is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Use vectorstore directly (public API)
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            return Response({"answer": "I could not find relevant information for your question."}, status=status.HTTP_200_OK)

        # Combine retrieved chunks
        context = "\n".join([d.page_content for d in docs])

        # Build prompt for Ollama
        prompt = f"""
        Use the context below to answer the question. Context:{context} Question:{query}"""

        # Generate answer
        answer = llm.invoke(prompt)

        return Response({"answer": answer}, status=status.HTTP_200_OK)

    except Exception as e:
        # logger.exception("An error occurred while processing the request")
        return Response({"error": "An error occurred while processing the request"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
