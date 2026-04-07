import os
from rag import RAGAgent

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"

agent = RAGAgent(
    api_key=API_KEY,
    model_name=MODEL,
    knowledge_dir="knowledge",
)

# Ingest all test docs
agent.add_dir("test_docs")

# Query
response = agent.respond("What is Python and how does it handle code blocks?")
print("Answer:", response.answer)
print("\nSources:", response.sources)

response = agent.respond("What is Python and how does it handle code blocks?")
print("Answer:", response.answer)
print("\nSources:", response.sources)

response = agent.respond("What is the key to a delicious chocolate cake?")
print("Answer:", response.answer)
print("\nSources:", response.sources)



