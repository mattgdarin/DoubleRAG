from .base import Judge, JudgeScore

DEFAULT_SYSTEM_PROMPT = """You are an impartial judge evaluating the ingestion quality of a RAG system.
You will be given the original source documents and the resulting knowledge base files after ingestion.
Score the ingestion on three dimensions from 1-5, then give an overall score.

Preservation (1-5): Was all information from the source documents retained?
Deduplication (1-5): Was redundant information eliminated without losing unique content?
Organisation (1-5): Is the resulting hierarchy logical and well-structured for the content?

Reply in this exact format:
Preservation: <score>
Deduplication: <score>
Organisation: <score>
Overall: <score>
Reasoning: <one or two sentences>"""


class IngestionJudge(Judge):
    def __init__(self, api_key: str, model_name: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        super().__init__(api_key, model_name, system_prompt)

    def score(self, source_docs: list[str], knowledge_files: list[str]) -> JudgeScore:
        sources_block = "\n\n---\n\n".join(source_docs)
        knowledge_block = "\n\n---\n\n".join(knowledge_files)

        raw = self._send([{"role": "user", "content": (
            f"Source documents:\n{sources_block}\n\n"
            f"Knowledge base files after ingestion:\n{knowledge_block}"
        )}])

        scores = {}
        reasoning = ""
        for line in raw.splitlines():
            for key in ("Preservation", "Deduplication", "Organisation", "Overall"):
                if line.startswith(f"{key}:"):
                    try:
                        scores[key] = round(float(line.split(":")[1].strip().split("/")[0].strip()))
                    except ValueError:
                        scores[key] = 0
            if line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        if "Overall" not in scores:
            dims = [scores[k] for k in ("Preservation", "Deduplication", "Organisation") if k in scores]
            scores["Overall"] = round(sum(dims) / len(dims)) if dims else 0

        return JudgeScore(score=scores.get("Overall", 0), reasoning=reasoning)
