import os
from rag import RAGAgent

MODEL = "claude-sonnet-4-6"

def query_mode(agent: RAGAgent):
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
        print(f"\nSources: {response.sources}\n")

def main():
    print("Welcome to DoubleRAG.")
    api_key = input("Enter your Anthropic API key: ").strip()
    agent = RAGAgent(api_key=api_key, model_name=MODEL, knowledge_dir="knowledge")
    while True:
        print("\n  1. Query")
        print("  2. Add file")
        print("  3. Add directory")
        print("  4. Exit")
        choice = input("\nChoice: ").strip()

        if choice == "1":
            query_mode(agent)

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
