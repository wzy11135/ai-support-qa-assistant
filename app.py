import json
from pathlib import Path

import streamlit as st

from config import KNOWLEDGE_BASE_DIR, OUTPUT_DIR
from qa.evaluator import run_evaluation
from qa.quality_checker import check_quality
from rag.generator import answer_question
from rag.loader import load_documents
from rag.splitter import split_documents
from rag.store_factory import build_retriever


st.set_page_config(page_title="AI Support QA Assistant", page_icon="AI", layout="wide")


@st.cache_resource
def build_vector_store():
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = split_documents(documents)
    vector_store, backend_name = build_retriever(chunks)
    return vector_store, len(documents), len(chunks), backend_name


vector_store, doc_count, chunk_count, backend_name = build_vector_store()

st.title("AI Support QA Assistant")
st.caption("基于知识库检索、引用溯源、客服质检和自动评估的 AI 工程项目")

metric_cols = st.columns(3)
metric_cols[0].metric("知识库文档", doc_count)
metric_cols[1].metric("检索片段", chunk_count)
metric_cols[2].metric("检索后端", backend_name)

tab_qa, tab_quality, tab_eval, tab_docs = st.tabs(["知识库问答", "客服质检", "评估报告", "知识库"])

with tab_qa:
    st.subheader("知识库问答")
    question = st.text_input("输入客户问题", value="超过7天还能申请退款吗？")
    top_k = st.slider("Top-K 检索数量", min_value=1, max_value=6, value=4)
    if st.button("生成回答", type="primary"):
        result = answer_question(question, vector_store, top_k=top_k)
        st.markdown("#### AI 回答")
        st.write(result.answer)
        st.caption(f"模型/模式：{result.used_provider}")
        st.markdown("#### 引用来源")
        for source in result.sources:
            with st.expander(f"{source.chunk.source} | score={source.score:.3f}"):
                st.write(source.chunk.text)

with tab_quality:
    st.subheader("客服对话质检")
    default_chat = """用户：我买的耳机第10天坏了，可以退吗？
客服：任何情况都可以全额退款，您放心，我们一定马上给您退。"""
    chat_text = st.text_area("粘贴客服对话", value=default_chat, height=180)
    if st.button("生成质检报告", type="primary"):
        report = check_quality(chat_text, vector_store)
        cols = st.columns(4)
        cols[0].metric("质检分", report.score)
        cols[1].metric("风险等级", report.risk_level)
        cols[2].metric("问题类型", report.issue_type)
        cols[3].metric("是否合规", "是" if report.is_compliant else "否")
        st.markdown("#### 质检原因")
        st.write(report.reason)
        st.markdown("#### 建议回复")
        st.write(report.suggested_reply)
        st.caption(f"模型/模式：{report.used_provider}")

with tab_eval:
    st.subheader("自动化评估")
    st.write("运行内置测试集，统计关键词命中率、引用命中率、拒答准确率和平均延迟。")
    if st.button("运行评估", type="primary"):
        report = run_evaluation(vector_store, Path("data/eval_questions.json"))
        OUTPUT_DIR.mkdir(exist_ok=True)
        report_path = OUTPUT_DIR / "eval_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        cols = st.columns(4)
        cols[0].metric("关键词命中率", report["keyword_match_rate"])
        cols[1].metric("引用命中率", report["source_hit_rate"])
        cols[2].metric("拒答准确率", report["refusal_accuracy"])
        cols[3].metric("平均延迟 ms", report["avg_latency_ms"])
        st.dataframe(report["cases"], use_container_width=True)
        st.success(f"评估报告已保存：{report_path}")

with tab_docs:
    st.subheader("知识库文档")
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        with st.expander(path.name):
            st.markdown(path.read_text(encoding="utf-8"))

