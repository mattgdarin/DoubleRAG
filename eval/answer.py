from .base import Judge, JudgeScore

DEFAULT_SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of a RAG system's answer.
You will be given a query, the context retrieved from the knowledge base, and the answer produced.
Score the answer on three dimensions from 1-5, then give an overall score.

Faithfulness (1-5): Is every claim in the answer supported by the context? Penalise hallucination.
Relevance (1-5): Does the answer actually address the query?
Completeness (1-5): Does the answer use the available context well, or miss important information?

Reply in this exact format:
Faithfulness: <score>
Relevance: <score>
Completeness: <score>
Overall: <score>
Reasoning: <one or two sentences>"""


class AnswerJudge(Judge):
    def __init__(self, api_key: str, model_name: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        super().__init__(api_key, model_name, system_prompt)

    def score(self, query: str, context: str, answer: str) -> JudgeScore:
        raw = self._send([{"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}"
        )}])

        scores = {}
        reasoning = ""
        for line in raw.splitlines():
            for key in ("Faithfulness", "Relevance", "Completeness", "Overall"):
                if line.startswith(f"{key}:"):
                    try:
                        scores[key] = round(float(line.split(":")[1].strip().split("/")[0].strip()))
                    except ValueError:
                        scores[key] = 0
            if line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        if "Overall" not in scores:
            dims = [scores[k] for k in ("Faithfulness", "Relevance", "Completeness") if k in scores]
            scores["Overall"] = round(sum(dims) / len(dims)) if dims else 0

        return JudgeScore(score=scores.get("Overall", 0), reasoning=reasoning)
