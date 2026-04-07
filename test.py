import os
from agent import IngestionAgent

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"

agent = IngestionAgent(
    api_key=API_KEY,
    model_name=MODEL,
    knowledge_dir="knowledge",
)

files = [
    "test_docs/python_overview_2.txt",
]

for f in files:
    print(f"Ingesting {f}...")
    agent.ingest(f)
    print(f"Done: {f}\n")
