# DoubleRAG

DoubleRAG is built on two main layers and a shared hierarchical database. The highest in the hierarchy is "topic," which describe the major topic of a piece of text, then "children" which describes the subtopic, and then the texts themselves.

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Document  │ ──▶ │ Ingestion Agent │ ──▶ │  Hierarchy  │
└─────────────┘     └─────────────────┘     └──────┬──────┘
                                                   │
┌─────────────┐     ┌─────────────────┐            │
│    Query    │ ──▶ │   Query Agent   │ ◀──────────┘
└─────────────┘     └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Answer      │
                    │   + Sources     │
                    └─────────────────┘
```

## Ingestion Layer
The first one is an ingestion layer. When the user wants to give a document for the RAG agent to use in the future. The ingestion layer chunks the text in a file, then for each chunk, it looks for the overarching topic, then it finds a subtopic, and then it looks at the documents within the subtopic.
If there is sufficient overlap, the agent will attempt to "merge" with an exists document in order to limit redundancy. If at any point, the model finds that the chunk does not fit into a certain topic/subtopic/file, it will create a new topic/subtopic/file and place it there.

## Query Agent
The query agent navigates the hierarchy to retrieve relevant context:

1. Identifies relevant topics based on the query
2. Identifies relevant children within those topics
3. Reads the matching files
4. Checks if context is sufficient to answer
5. Falls back to web search if insufficient (with optional ingestion)
6. Generates answer with source citations

The agent also learns from conversations — extracting useful facts for future retrieval and inferring user preferences to adapt responses.


## What It Does

- **Ingestion Agent**: Organizes documents into a navigable topic/subtopic hierarchy, merging related content and eliminating redundancy
- **Query Agent**: Navigates the hierarchy, checks sufficiency, falls back to web search, learns from conversations
- **Evaluation**: LLM-as-judge scoring for retrieval, answer quality, and ingestion

## How To Use

This agent is mainly expected to be used as a library.

### Installation

```bash
git clone https://github.com/mattgdarin/DoubleRAG.git
cd DoubleRAG
pip install -r requirements.txt
cp .env.example .env  # Add your API key
```

### As a Library

```python
from agent.rag import RAGAgent

agent = RAGAgent(
    api_key="your-api-key",
    model_name="claude-sonnet-4-20250514",
    knowledge_dir="./knowledge_base"
)

# Ingest documents
agent.add_file("path/to/document.pdf")
agent.add_dir("path/to/docs/")

# Query
response = agent.respond("What is X?")
print(response.answer)
print(response.sources)
```
### Demo files

```bash
python main.py
```

This launches an interactive menu where you can:
1. Query the knowledge base
2. Add a file
3. Add a directory
4. Exit

## Evaluation

DoubleRAG includes an LLM-as-judge evaluation framework that benchmarks against a vanilla RAG baseline (chunking + vector search via ChromaDB).

### Running the eval

```bash
python3 -m eval.compare
```

This will:
1. Ingest the test documents into both DoubleRAG and VanillaRAG
2. Run a set of queries against both systems
3. Score each response using three judges: `AnswerJudge`, `SourceJudge`, and `IngestionJudge`
4. Print a summary table and per-query breakdown

### Test documents

Place test documents in `test_docs/`. The eval currently uses a fictional company dataset (`test_docs/acme/`) with two documents covering company overview, financials, competitors, HR, and upcoming initiatives.

To run a meaningful comparison, use documents with real overlap and redundancy — this stresses DoubleRAG's merge/dedup logic, which is its primary differentiator.

### Early results

Tested on 12 queries across simple factual, cross-document reasoning, inference, multi-hop, and negation question types:

| Metric | DoubleRAG | VanillaRAG |
|---|---|---|
| Answer quality (avg /5) | 4.50 | 4.58 |
| Source quality (avg /5) | 3.42 | 3.33 |

Results are near-identical on a small 2-document dataset, which is expected — DoubleRAG's hierarchy and deduplication advantages emerge at scale with overlapping documents. A larger dataset with meaningful redundancy is the next step for evaluation. 
