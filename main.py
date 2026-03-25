import uvicorn
from fastapi import FastAPI, UploadFile
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, SecretStr
import logging

import traceback
import os

from dotenv import load_dotenv


app = FastAPI(title="RAG-Chroma")


# 设置日志级别为INFO,修改默认的日志级别，不然的话info级别的日志不会被输出
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-system")


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("缺少环境变量 OPENAI_API_KEY")

openai_base_url = os.getenv("OPENAI_BASE_URL")
if not openai_base_url:
    raise ValueError("缺少环境变量 OPENAI_BASE_URL")

openai_model = os.getenv("OPENAI_MODEL")
if not openai_model:
    raise ValueError("缺少环境变量 OPENAI_MODEL")

ollama_base_url = os.getenv("OLLAMA_BASE_URL")
if not ollama_base_url:
    raise ValueError("缺少环境变量 OLLAMA_BASE_URL")

ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
if not ollama_embedding_model:
    raise ValueError("缺少环境变量 OLLAMA_EMBEDDING_MODEL")


COLLECTION_NAME = "jiejie_test_v2"
# 2. 设置 Chroma 向量数据库在本地保存的文件夹路径
CHROMA_PERSIST_DIR = "./chroma_data"

# 初始化 Embedding 模型
embeddings = OllamaEmbeddings(
    model=ollama_embedding_model,
    base_url=ollama_base_url,
    client_kwargs={"trust_env": False},
)

chatModel = ChatOpenAI(
    model=openai_model,
    api_key=SecretStr(openai_api_key),
    base_url=openai_base_url,  # 👈 重点：这里加上 /v1
)


# 3. 初始化 Chroma 向量数据库
# 它会自动在当前目录下创建一个 chroma_data 文件夹来持久化保存你的向量
vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)

# 配置文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


@app.get("/text_embedding")
def get_text_embedding(text: str):
    try:
        logger.info("收到 text_embedding 请求，text_length=%s", len(text))
        vector = embeddings.embed_query(text)
        return {"embedding": vector}
    except Exception as e:
        logger.exception("text_embedding 调用失败")
        return {"error": str(e)}


@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        logger.info("收到 upload 请求，filename=%s", file.filename)
        file_content = await file.read()

        # 兼容不同编码格式的纯文本文件
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_content.decode("gbk")
            except UnicodeDecodeError:
                return {"error": "文件解码失败！请确保上传的是纯文本文件 (如 .txt)。"}

        if not text.strip():
            return {"error": "上传的文件内容为空"}

        # 将长文本切块
        chunks = text_splitter.split_text(text)
        if not chunks:
            return {"error": "文本切块后无有效内容"}

        documents = [
            Document(page_content=chunk, metadata={"filename": file.filename})
            for chunk in chunks
        ]

        # 4. 向量入库 (Chroma 会自动处理并保存到 persist_directory)
        try:
            vector_db.add_documents(documents)
            logger.info("上传完成，filename=%s, chunks=%s", file.filename, len(chunks))
        except Exception as db_err:
            logger.exception("Chroma 入库失败，filename=%s", file.filename)
            return {"error": f"向量入库失败: {str(db_err)}"}

        return {"status": "success", "chunks_count": len(chunks)}

    except Exception as e:
        logger.exception("upload 接口发生未知错误，filename=%s", file.filename)
        return {"error": f"接口发生未知错误: {str(e)}"}


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    try:
        logger.info("收到 chat 请求，query=%s", query)

        # 1. 从 Chroma 数据库中检索最相关的 Top-3 文档块
        # 这里的 k=3 表示返回 3 个最相关的 chunk
        docs = vector_db.similarity_search(query, k=3)

        logger.info("检索到 %s 个相关文档", len(docs))
        for doc in docs:
            logger.info("相关文档: %s", doc.metadata.get("filename", "未知文件"))

        # 2. 【第一重保险】：如果数据库里完全没有数据，直接用大模型本身的能力
        if not docs:
            logger.info("知识库为空，直接调用大模型")
            response = chatModel.invoke(query)
            return {
                "answer": response.content,
                "source": "llm_fallback",
                "referenced_files": [],
            }

        # 将检索到的文档内容拼接在一起，作为上下文
        context = "\n\n".join([doc.page_content for doc in docs])

        logger.info("context=%s", context)

        # 提取相关文档的文件名，方便前端展示引用来源
        source_files = list(
            set([doc.metadata.get("filename", "未知文件") for doc in docs])
        )

        # 3. 【第二重保险】：构建 Prompt
        # 核心逻辑：明确告诉大模型，参考资料可能没用，没用就自己作答
        prompt_template = """你是一个智能助手。请优先使用以下【参考资料】来回答用户的问题。
        如果【参考资料】中没有包含能回答该问题的信息，请忽略参考资料，直接使用你自己的知识来回答。
        请直接给出回答，不要在回答中说明“参考资料中没有提及”或“根据我的知识”。

        【参考资料】：
        {context}

        【用户问题】：{query}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.invoke({"context": context, "query": query})

        # 4. 构建 LangChain 运行链并执行
        chain = prompt | chatModel | StrOutputParser()

        answer = chain.invoke({"context": context, "query": query})

        logger.info(
            "chat 完成，source=rag_or_llm, referenced_files=%s",
            source_files,
        )

        return {
            "answer": answer,
            "source": "rag_or_llm",  # 因为是模型自行判断的，所以统一返回状态
            "referenced_files": source_files,  # 返回检索到的文件，即使大模型没用上，也可以展示出来
        }

    except Exception as e:
        logger.exception("chat 接口发生错误，query=%s", query)
        return {"error": f"对话接口发生错误: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="info", access_log=True)
