# AI Support QA Assistant

基于 RAG 的客服知识库问答与客服质检系统。项目面向真实客服场景：把售后政策、物流规则、会员 FAQ 和质保说明构造成知识库，支持用户问题回答、引用溯源、客服对话质检和自动化评估。

这个项目适合写进 AI 工程师简历，因为它不仅是聊天页面，还包含检索增强生成、业务质检、结构化输出、评估指标和可演示页面。

## 核心功能

- 知识库问答：根据本地 Markdown 知识库回答客户问题。
- 引用溯源：展示 Top-K 检索片段、来源文档和相似度分数。
- 客服质检：识别事实错误、遗漏信息、语气问题和过度承诺。
- 自动化评估：统计关键词命中率、引用命中率、拒答准确率和平均延迟。
- 低配置可运行：默认使用本地轻量检索，接入 LLM API 后可升级回答质量。

## 技术栈

- Python
- Streamlit
- 本地向量检索 / RAG Pipeline
- OpenAI-compatible LLM API，可接 DeepSeek、OpenAI、通义千问等兼容接口
- JSON 测试集与自动评估报告

## 项目结构

```text
ai-support-qa-assistant/
  app.py
  config.py
  services.py
  rag/
    loader.py
    splitter.py
    retriever.py
    generator.py
  qa/
    quality_checker.py
    evaluator.py
  data/
    knowledge_base/
    eval_questions.json
    sample_chats.json
  tests/
  README.md
```

## 快速启动

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

没有 API Key 也可以运行，系统会使用本地兜底模式完成演示。

## 配置 LLM API

复制 `.env.example` 为 `.env`，填入兼容 OpenAI Chat Completions 格式的 API。

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

## RAG 流程

```text
知识库文档
  -> 文档切分
  -> 本地向量化索引
  -> 用户问题检索 Top-K
  -> 构造上下文 Prompt
  -> LLM 生成答案
  -> 输出答案和引用来源
```

## 评估指标

- `keyword_match_rate`：回答是否命中预期关键词。
- `source_hit_rate`：回答是否检索到知识库来源。
- `refusal_accuracy`：无关问题是否正确拒答。
- `avg_latency_ms`：平均响应延迟。

## 简历写法参考

项目：AI Support QA Assistant - 基于 RAG 的客服质检与知识库问答系统

- 基于 Python、Streamlit 和 LLM API 构建客服 AI 辅助平台，支持知识库问答、引用溯源、客服对话质检和自动化评估。
- 实现 RAG 检索链路，包括文档加载、chunking、本地向量检索、Top-K 上下文召回和基于来源的回答生成，降低无依据回答风险。
- 设计客服质检模块，自动识别客服回复中的事实错误、遗漏信息、语气问题和过度承诺，并输出结构化质检报告。
- 构建评估集统计关键词命中率、引用命中率、拒答准确率和平均响应延迟，用于评估检索和生成质量。

## 后续可升级方向

- 将本地检索替换为 ChromaDB、FAISS 或 pgvector。
- 增加 rerank 模块，提高长文档检索准确率。
- 增加 FastAPI 后端和 React 前端。
- 使用 Docker Compose 一键部署。
- 接入日志系统，统计 token 成本、延迟和用户反馈。

