from .base import Judge, JudgeScore

DEFAULT_SYSTEM_PROMPT = """You are an impartial judge evaluating the retrieval quality of a RAG system.
You will be given a query, the knowledge base index, and the sources that were retrieved.
Score the retrieval on two dimensions from 1-5, then give an overall score.

Precision (1-5): Are the retrieved sources relevant to the query? Penalise irrelevant retrievals.
Recall (1-5): Were important sources missed that should have been retrieved?

Reply in this exact format:
Precision: <score>
Recall: <score>
Overall: <score>
Reasoning: <one or two sentences>"""


class SourceJudge(Judge):
    def __init__(self, api_key: str, model_name: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        super().__init__(api_key, model_name, system_prompt)

    def score(self, query: str, index: dict, sources: list[str]) -> JudgeScore:
        raw = self._send([{"role": "user", "content": (
            f"Query: {query}\n\n"
            f"Knowledge base index:\n{index}\n\n"
            f"Retrieved sources:\n{chr(10).join(sources)}"
        )}])

        scores = {}
        reasoning = ""
        for line in raw.splitlines():
            for key in ("Precision", "Recall", "Overall"):
                if line.startswith(f"{key}:"):
                    try:
                        scores[key] = round(float(line.split(":")[1].strip().split("/")[0].strip()))
                    except ValueError:
                        scores[key] = 0
            if line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        if "Overall" not in scores:
            dims = [scores[k] for k in ("Precision", "Recall") if k in scores]
            scores["Overall"] = round(sum(dims) / len(dims)) if dims else 0

        return JudgeScore(score=scores.get("Overall", 0), reasoning=reasoning)
