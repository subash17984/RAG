from langchain_community.llms import Ollama

llm=Ollama(model="gpt-oss:120b-cloud", base_url="http://localhost:11434")
