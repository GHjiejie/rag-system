from fastapi import FastAPI, UploadFile
from langchain_ollama import OllamaEmbeddings
import uvicorn
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

# 1. 引入 Chroma
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import traceback
import os


app = FastAPI(title="RAG-Chroma")

COLLECTION_NAME = "jiejie_test_v2"
# 2. 设置 Chroma 向量数据库在本地保存的文件夹路径
CHROMA_PERSIST_DIR = "./chroma_data"

# 初始化 Embedding 模型
embeddings = OllamaEmbeddings(
    model="bge-m3:567m",
    base_url="http://localhost:11434",
    client_kwargs={"trust_env": False},
)

chatModel = ChatOpenAI(
    model="gpt-5.4",
    api_key=SecretStr(
        "sk-846f1b178fbb2f6810f0cda48c69c9a7c2ffd29c8847887820bf7ee0a17a8945"
    ),
    # api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
    base_url="https://gmncode.cn/v1",  # 👈 重点：这里加上 /v1
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
        vector = embeddings.embed_query(text)
        return {"embedding": vector}
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
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
            print(f"成功将 {len(chunks)} 个数据块存入 ChromaDB!")
        except Exception as db_err:
            print("=== Chroma 入库报错 ===")
            traceback.print_exc()
            return {"error": f"向量入库失败: {str(db_err)}"}

        return {"status": "success", "chunks_count": len(chunks)}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"接口发生未知错误: {str(e)}"}


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query

    try:
        # 1. 从 Chroma 数据库中检索最相关的 Top-3 文档块
        # 这里的 k=3 表示返回 3 个最相关的 chunk
        docs = vector_db.similarity_search(query, k=3)

        # 2. 【第一重保险】：如果数据库里完全没有数据，直接用大模型本身的能力
        if not docs:
            print("知识库为空，直接调用大模型...")
            response = chatModel.invoke(query)
            return {
                "answer": response.content,
                "source": "llm_fallback",
                "referenced_files": [],
            }

        # 将检索到的文档内容拼接在一起，作为上下文
        context = "\n\n".join([doc.page_content for doc in docs])

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

        # 4. 构建 LangChain 运行链并执行
        chain = prompt | chatModel | StrOutputParser()

        answer = chain.invoke({"context": context, "query": query})

        return {
            "answer": answer,
            "source": "rag_or_llm",  # 因为是模型自行判断的，所以统一返回状态
            "referenced_files": source_files,  # 返回检索到的文件，即使大模型没用上，也可以展示出来
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"对话接口发生错误: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
