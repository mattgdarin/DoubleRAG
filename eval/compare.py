import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from rag import RAGAgent
from vanilla_rag import VanillaRAG
from eval import AnswerJudge, SourceJudge

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheaper model for judging
DOCS_DIR = "test_docs/acme"
KNOWLEDGE_DIR = "knowledge_eval"

QUERIES = [
    # Simple factual
    "Who founded Acme Corp and when?",
    "What is the ClearFlow 3000 and how does it work?",
    "What percentage of Acme's revenue comes from maintenance contracts?",
    "What benefits does Acme Corp offer its employees?",

    # Cross-document reasoning
    "How does Acme's R&D team and product roadmap support their competitive differentiation against PureWave?",
    "How do Acme's hiring practices and employee retention relate to their operational performance?",

    # Inference
    "Based on their expansion plans and current markets, what regions is Acme targeting by 2026?",
    "What does Acme's revenue mix suggest about the stability of their business model?",

    # Multi-hop
    "Who leads the team responsible for developing AquaWatch, and what is their role at Acme?",
    "Which executive joined from the firm that led Acme's most recent funding round?",

    # Negation / absence
    "Does Acme Corp currently operate in any markets outside North America?",
    "Does Acme manufacture its own activated carbon media, or does it source it externally?",
]


@dataclass
class ComparisonResult:
    query: str
    double_answer: str
    vanilla_answer: str
    double_sources: list[str]
    vanilla_sources: list[str]
    double_answer_score: int
    vanilla_answer_score: int
    double_answer_reasoning: str
    vanilla_answer_reasoning: str
    double_source_score: int
    vanilla_source_score: int


def run_comparison():
    print("Setting up DoubleRAG...")
    double_agent = RAGAgent(
        api_key=API_KEY,
        model_name=MODEL,
        knowledge_dir=KNOWLEDGE_DIR,
    )
    double_agent.add_dir(DOCS_DIR)

    print("Setting up VanillaRAG...")
    vanilla_agent = VanillaRAG(
        api_key=API_KEY,
        model_name=MODEL,
        collection_name="eval_collection",
    )
    vanilla_agent.add_dir(DOCS_DIR)

    answer_judge = AnswerJudge(api_key=API_KEY, model_name=JUDGE_MODEL)
    source_judge = SourceJudge(api_key=API_KEY, model_name=JUDGE_MODEL)

    results: list[ComparisonResult] = []

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] Query: {query}")

        print("  DoubleRAG...")
        double_response = double_agent.respond(query)

        print("  VanillaRAG...")
        vanilla_response = vanilla_agent.respond(query)

        print("  Judging answers...")
        from pathlib import Path
        double_context = "\n\n---\n\n".join(
            Path(KNOWLEDGE_DIR, s).read_text(encoding="utf-8")
            for s in double_response.sources
            if Path(KNOWLEDGE_DIR, s).exists()
        )
        double_answer_score = answer_judge.score(
            query=query,
            context=double_context,
            answer=double_response.answer,
        )
        vanilla_answer_score = answer_judge.score(
            query=query,
            context=vanilla_agent._last_context,
            answer=vanilla_response.answer,
        )

        print("  Judging sources...")
        double_source_score = source_judge.score(
            query=query,
            index=double_response.sources,
            sources=double_response.sources,
        )
        vanilla_source_score = source_judge.score(
            query=query,
            index=vanilla_response.sources,
            sources=vanilla_response.sources,
        )

        results.append(ComparisonResult(
            query=query,
            double_answer=double_response.answer,
            vanilla_answer=vanilla_response.answer,
            double_sources=double_response.sources,
            vanilla_sources=vanilla_response.sources,
            double_answer_score=double_answer_score.score,
            vanilla_answer_score=vanilla_answer_score.score,
            double_answer_reasoning=double_answer_score.reasoning,
            vanilla_answer_reasoning=vanilla_answer_score.reasoning,
            double_source_score=double_source_score.score,
            vanilla_source_score=vanilla_source_score.score,
        ))

    print_summary(results)
    return results


def print_summary(results: list[ComparisonResult]):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    double_answer_total = sum(r.double_answer_score for r in results)
    vanilla_answer_total = sum(r.vanilla_answer_score for r in results)
    double_source_total = sum(r.double_source_score for r in results)
    vanilla_source_total = sum(r.vanilla_source_score for r in results)
    n = len(results)

    print(f"\n{'Metric':<25} {'DoubleRAG':>10} {'VanillaRAG':>12}")
    print("-" * 50)
    print(f"{'Answer quality (avg)':<25} {double_answer_total/n:>10.2f} {vanilla_answer_total/n:>12.2f}")
    print(f"{'Source quality (avg)':<25} {double_source_total/n:>10.2f} {vanilla_source_total/n:>12.2f}")

    print("\nPER-QUERY BREAKDOWN")
    print("-" * 50)
    for r in results:
        winner = "DOUBLE" if r.double_answer_score > r.vanilla_answer_score else \
                 "VANILLA" if r.vanilla_answer_score > r.double_answer_score else "TIE"
        print(f"\nQ: {r.query[:60]}...")
        print(f"  Answer:  DoubleRAG={r.double_answer_score}/5  VanillaRAG={r.vanilla_answer_score}/5  → {winner}")
        print(f"  Sources: DoubleRAG={r.double_source_score}/5  VanillaRAG={r.vanilla_source_score}/5")
        print(f"  Double reasoning:  {r.double_answer_reasoning}")
        print(f"  Vanilla reasoning: {r.vanilla_answer_reasoning}")


if __name__ == "__main__":
    run_comparison()
