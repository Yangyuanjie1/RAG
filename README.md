# RAG 知识问答系统

大模型应用项目 它实现了一个完整的 RAG 最小闭环：文档上传、文本切块、向量化、相似度检索、基于上下文生成答案。

## 项目功能

- 支持上传 `.txt`、`.md`、`.pdf` 文档
- 对文档进行本地切块
- 使用 embedding 模型为文档分块生成向量
- 通过余弦相似度召回最相关内容
- 将召回结果与问题一起发送给 MiniMax
- 在页面中展示答案和来源片段

## 技术栈

- Python
- FastAPI
- MiniMax OpenAI 兼容接口
- sentence-transformers
- BAAI/bge-small-zh-v1.5
- NumPy
- PyPDF
- HTML / CSS / JavaScript

## 工作流程

1. 用户上传文档
2. 系统读取文档内容
3. 将文档切分为多个 chunk
4. 为每个 chunk 生成 embedding
5. 用户提出问题
6. 系统将问题转成 embedding
7. 通过向量相似度检索最相关 chunk
8. 将问题和检索结果一起发送给 MiniMax
9. 返回答案并展示来源

## 项目结构

```text
project-02-rag-assistant
├─ app
│  ├─ api
│  ├─ core
│  ├─ schemas
│  └─ services
├─ data
│  └─ uploads
├─ static
│  └─ app.html
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## 本地运行

```powershell
cd D:\codex1\project-02-rag-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8001
```

启动后访问：

- 首页：`http://127.0.0.1:8001`
- 接口文档：`http://127.0.0.1:8001/docs`
<img width="2535" height="1393" alt="image" src="https://github.com/user-attachments/assets/93e17569-88be-47aa-b74f-52af646c7605" />



## 环境变量

```env
APP_NAME=RAG Knowledge Assistant
API_PREFIX=/api
OPENAI_API_KEY=your_minimax_api_key_here
OPENAI_BASE_URL=https://api.minimaxi.com/v1
OPENAI_MODEL=MiniMax-M2.5
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHUNK_SIZE=500
CHUNK_OVERLAP=80
RETRIEVAL_K=3
```

## 当前版本说明

这版已经是向量检索版 RAG，但向量数据仍保存在内存中，适合作为学习项目和作品集项目。

如果继续升级，建议优先做：

1. 接入 FAISS 或 Chroma 做持久化向量库
2. 增加文档元数据管理
3. 展示真实返回模型名
4. 增加流式输出


