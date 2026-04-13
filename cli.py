import os
from dotenv import load_dotenv
load_dotenv()
import sys
from rag import RAGAgent

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"

agent = RAGAgent(
    api_key=API_KEY,
    model_name=MODEL,
    knowledge_dir="knowledge",
)

#agent.add_dir("test_docs")

def ingest(path: str) -> None:
    if os.path.isdir(path):
        print(f"Ingesting directory: {path}")
        agent.add_dir(path)
    elif os.path.isfile(path):
        print(f"Ingesting file: {path}")
        agent.add_file(path)
    else:
        print(f"[error] Path not found: {path}")
        return
    print("Ingestion complete.\n")

def chat() -> None:
    print("DoubleRAG — type 'exit' to quit, 'ingest <path>' to add documents.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break
        if user_input.lower().startswith("ingest "):
            ingest(user_input[7:].strip())
            continue

        response = agent.respond(user_input)
        print(f"\nSources: {response.answer}\n")
        print(f"\nSources: {response.sources}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ingest(sys.argv[1])
    chat()
