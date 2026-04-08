import os
from rag import RAGAgent

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"

agent = RAGAgent(
    api_key=API_KEY,
    model_name=MODEL,
    knowledge_dir="knowledge",
)

def query_mode():
    print("\nEntering query mode. Type 'exit' to return to the main menu.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() == "exit":
            break
        response = agent.respond(query)
        print(f"\Answer: {response.answer}\n")
        print(f"\nSources: {response.sources}\n")

def main():
    print("Welcome to DoubleRAG.")
    while True:
        print("\n  1. Query")
        print("  2. Add file")
        print("  3. Add directory")
        print("  4. Exit")
        choice = input("\nChoice: ").strip()

        if choice == "1":
            query_mode()

        elif choice == "2":
            path = input("File path: ").strip()
            if not os.path.isfile(path):
                print(f"[error] File not found: {path}")
                continue
            print(f"Ingesting {path}...")
            agent.add_file(path)
            print("Done.")

        elif choice == "3":
            path = input("Directory path: ").strip()
            if not os.path.isdir(path):
                print(f"[error] Directory not found: {path}")
                continue
            print(f"Ingesting {path}...")
            agent.add_dir(path)
            print("Done.")

        elif choice == "4":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
