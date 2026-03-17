# agent-demos
LLM & Agent示例

## 依赖
Python 3.12

安装依赖库：

```shell
pip install -r requirements.txt
```

## 运行
示例使用DeepSeek作为LLM，需要将环境变量`DEEPSEEK_API_KEY`设置为API key。

```shell
export DEEPSEEK_API_KEY=sk-xxx
cd langchain
python chat.py 
```

## 示例代码
* [调用DeepSeek API](deepseek_api)
  * [简单对话](deepseek_api/chat.py)
* [LangChain框架](langchain)
  * [简单对话](langchain/chat.py)：LLM集成
  * [天气预报Agent](langchain/tool_call/weather_agent.py)：Agent+工具调用
  * [SQL Agent](langchain/sql_agent/sql_agent.py)：SQL查询Agent
  * [PDF搜索引擎](langchain/knowledge_base/pdf_search_engine.py)：构建知识库+语义搜索
  * [网页检索Agent](langchain/rag/webpage_retrieval_agent.py)：Agentic RAG
  * [网站内容问答聊天机器人](langchain/rag/website_qa_chatbot.py)：RAG Agent
