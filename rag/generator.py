from dataclasses import dataclass

from config import MIN_RETRIEVAL_SCORE, TOP_K
from rag.retriever import LocalVectorStore, SearchResult
from services import LLMClient


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    sources: list[SearchResult]
    used_provider: str
    refused: bool


SYSTEM_PROMPT = """你是企业客服知识库助手。你必须只基于提供的知识库片段回答。
如果知识库没有依据，明确说无法根据当前知识库确认。回答要简洁、可执行，并保留必要条件。"""


def build_context(results: list[SearchResult]) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] 来源: {result.chunk.source}, score={result.score:.3f}\n{result.chunk.text}")
    return "\n\n".join(lines)


def local_answer(question: str, results: list[SearchResult]) -> str:
    if not results or results[0].score < MIN_RETRIEVAL_SCORE:
        return "无法根据当前知识库确认该问题，请补充相关政策文档或转人工处理。"

    evidence = results[0].chunk.text.replace("\n", " ")
    if len(evidence) > 260:
        evidence = evidence[:260] + "..."
    return f"根据知识库，{evidence}\n\n建议以引用来源为准，不要做超出政策范围的承诺。"


def answer_question(question: str, vector_store: LocalVectorStore, top_k: int = TOP_K) -> Answer:
    results = vector_store.search(question, top_k=top_k)
    refused = not results or results[0].score < MIN_RETRIEVAL_SCORE
    client = LLMClient()

    if refused:
        return Answer(
            question=question,
            answer="无法根据当前知识库确认该问题，请补充相关政策文档或转人工处理。",
            sources=results,
            used_provider="retrieval-guardrail",
            refused=True,
        )

    context = build_context(results)
    user_prompt = f"""问题：{question}

知识库片段：
{context}

请输出：
1. 直接答案
2. 注意事项
3. 引用来源编号"""

    response = client.chat(SYSTEM_PROMPT, user_prompt)
    if response.content:
        return Answer(question=question, answer=response.content, sources=results, used_provider=response.used_provider, refused=False)

    return Answer(question=question, answer=local_answer(question, results), sources=results, used_provider=response.used_provider, refused=False)

