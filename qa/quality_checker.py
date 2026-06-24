import json
import re
from dataclasses import dataclass

from rag.generator import build_context
from rag.retriever import LocalVectorStore
from services import LLMClient


@dataclass(frozen=True)
class QualityReport:
    score: int
    issue_type: str
    risk_level: str
    is_compliant: bool
    reason: str
    suggested_reply: str
    used_provider: str


SYSTEM_PROMPT = """你是客服质检专家。你需要根据知识库判断客服回复是否准确、合规、礼貌。
必须输出 JSON，不要输出多余解释。字段：score, issue_type, risk_level, is_compliant, reason, suggested_reply。"""


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def local_quality_check(chat_text: str, vector_store: LocalVectorStore) -> QualityReport:
    risky_terms = ["任何情况", "保证", "一定", "全额退款", "永久", "无需审核"]
    has_risk = any(term in chat_text for term in risky_terms)
    results = vector_store.search(chat_text, top_k=3)
    score = 62 if has_risk else 86
    issue_type = "过度承诺" if has_risk else "未发现明显问题"
    risk_level = "high" if has_risk else "low"
    reason = "客服回复中出现绝对化承诺，可能与售后、退款或质保政策冲突。" if has_risk else "客服回复整体与知识库相关政策没有明显冲突。"
    if results:
        reason += f" 参考来源：{results[0].chunk.source}。"
    suggested_reply = "建议改为基于政策条件回复，并提醒用户以审核结果和订单状态为准。"
    return QualityReport(score, issue_type, risk_level, not has_risk, reason, suggested_reply, "local-fallback")


def check_quality(chat_text: str, vector_store: LocalVectorStore) -> QualityReport:
    results = vector_store.search(chat_text, top_k=4)
    client = LLMClient()
    context = build_context(results)
    user_prompt = f"""客服对话：
{chat_text}

相关知识库：
{context}

请判断客服是否存在事实错误、遗漏信息、语气问题、过度承诺或需要人工复核。"""

    response = client.chat(SYSTEM_PROMPT, user_prompt)
    if response.content:
        try:
            data = extract_json(response.content)
            return QualityReport(
                score=int(data["score"]),
                issue_type=str(data["issue_type"]),
                risk_level=str(data["risk_level"]),
                is_compliant=bool(data["is_compliant"]),
                reason=str(data["reason"]),
                suggested_reply=str(data["suggested_reply"]),
                used_provider=response.used_provider,
            )
        except Exception:
            pass

    return local_quality_check(chat_text, vector_store)

