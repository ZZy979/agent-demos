# agent-demos
LLM & Agent示例

## 依赖
Python 3.12

安装依赖库：

```shell
pip install -r requirements.txt
```

## 示例代码
* [调用DeepSeek API](deepseek_api)
  * [简单对话](deepseek_api/chat.py)
* [LangChain框架](langchain)
  * [简单对话](langchain/chat.py)：LLM集成
  * [天气预报Agent](langchain/weather_agent.py)：Agent+工具调用
  * [SQL Agent](langchain/sql_agent/sql_agent.py)：SQL查询Agent
  * [网页检索Agent](langchain/webpage_retrieval_agent.py)：Agentic RAG
  * [PDF搜索引擎](langchain/knowledge_base/pdf_search_engine.py)：构建知识库+语义搜索
