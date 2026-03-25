import logging
import os
from functools import lru_cache
from typing import Any

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, SecretStr

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少环境变量 {name}")
    return value


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return float(value)


def detect_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-system")

OPENAI_API_KEY = require_env("OPENAI_API_KEY")
OPENAI_BASE_URL = require_env("OPENAI_BASE_URL")
OPENAI_MODEL = require_env("OPENAI_MODEL")
OLLAMA_BASE_URL = require_env("OLLAMA_BASE_URL")
OLLAMA_EMBEDDING_MODEL = require_env("OLLAMA_EMBEDDING_MODEL")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "jiejie_test_v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
CHUNK_SIZE = env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = env_int("CHUNK_OVERLAP", 50)

RERANK_ENABLED = env_flag("RERANK_ENABLED", True)
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", detect_best_device())
RERANK_USE_FP16 = env_flag("RERANK_USE_FP16", RERANK_DEVICE == "cuda")
RETRIEVAL_TOP_K = max(1, env_int("RETRIEVAL_TOP_K", 6))
RERANK_TOP_N = max(1, env_int("RERANK_TOP_N", 3))
RERANK_BATCH_SIZE = max(1, env_int("RERANK_BATCH_SIZE", 32))
RERANK_SCORE_THRESHOLD = env_float("RERANK_SCORE_THRESHOLD")

logger.info(
    "服务启动配置: collection=%s, chroma_dir=%s, rerank_enabled=%s, rerank_model=%s, rerank_device=%s",
    COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    RERANK_ENABLED,
    RERANK_MODEL_NAME,
    RERANK_DEVICE,
)

app = FastAPI(title="RAG-Chroma")

embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
    client_kwargs={"trust_env": False},
)

chat_model = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=SecretStr(OPENAI_API_KEY),
    base_url=OPENAI_BASE_URL,
)

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# 加载 reranker模型
@lru_cache(maxsize=1)
def get_reranker():
    if not RERANK_ENABLED:
        logger.info("Rerank 已关闭，将仅使用向量召回结果。")
        return None

    if FlagReranker is None:
        logger.warning("未安装 FlagEmbedding，无法启用 rerank，将退回向量召回。")
        return None

    try:
        logger.info(
            "开始加载 reranker: model=%s, device=%s, fp16=%s, batch_size=%s",
            RERANK_MODEL_NAME,
            RERANK_DEVICE,
            RERANK_USE_FP16,
            RERANK_BATCH_SIZE,
        )
        return FlagReranker(
            RERANK_MODEL_NAME,
            use_fp16=RERANK_USE_FP16,
            devices=RERANK_DEVICE,
            batch_size=RERANK_BATCH_SIZE,
        )
    except Exception:
        logger.exception("加载 reranker 失败，将退回向量召回。")
        return None


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int,
) -> tuple[list[Document], list[dict[str, Any]], bool]:
    if not documents:
        return [], [], False

    reranker = get_reranker()
    if reranker is None:
        return documents[:top_n], [], False

    try:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.compute_score(pairs)
        logger.info("rerank 计算完成，scores=%s", scores)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [scores]
    except Exception:
        logger.exception("rerank 计算失败，将退回向量召回顺序。")
        return documents[:top_n], [], False

    if len(scores) != len(documents):
        logger.warning(
            "rerank 分数数量异常: scores=%s, documents=%s，将退回向量召回顺序。",
            len(scores),
            len(documents),
        )
        return documents[:top_n], [], False

    # 创建一个float类型的数组
    normalized_scores: list[float] = []
    for score in scores:
        # 判断score是不是int或者float其中的任意一个类型
        if isinstance(score, (int, float)):
            normalized_scores.append(float(score))
            continue

        if score is None:
            logger.warning("rerank 返回了 None 分数，将使用最低分处理。")
            normalized_scores.append(float("-inf"))
            continue

        if hasattr(score, "item"):
            try:
                normalized_scores.append(float(score.item()))
                continue
            except Exception:
                logger.warning("rerank 分数 item() 转换失败，将使用最低分处理。")

        try:
            normalized_scores.append(float(score))
        except (TypeError, ValueError):
            logger.warning("rerank 分数类型异常(%s)，将使用最低分处理。", type(score))
            normalized_scores.append(float("-inf"))

    # 一下是python的列表推导式语法
    scored_results = [
        {
            "score": score,
            "document": doc,
            "filename": doc.metadata.get("filename", "未知文件"),
        }
        for score, doc in zip(normalized_scores, documents)
    ]

    # 以上代码等价于传统写法
    #     scored_results = []
    # for score, doc in zip(normalized_scores, documents):
    #     scored_results.append(
    #         {
    #             "score": score,
    #             "document": doc,
    #             "filename": doc.metadata.get("filename", "未知文件"),
    #         }
    #     )

    logger.info("rerank 结果: %s", scored_results)
    scored_results.sort(key=lambda item: item["score"], reverse=True)

    if RERANK_SCORE_THRESHOLD is not None:
        selected_docs = [
            item["document"]
            for item in scored_results
            if item["score"] >= RERANK_SCORE_THRESHOLD
        ][:top_n]
    else:
        selected_docs = [item["document"] for item in scored_results[:top_n]]

    if not selected_docs:
        selected_docs = [scored_results[0]["document"]]

    return selected_docs, scored_results, True


def build_context(documents: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)


def unique_source_files(documents: list[Document]) -> list[str]:
    return sorted(
        {doc.metadata.get("filename", "未知文件") for doc in documents if doc.metadata}
    )


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rerank_enabled": RERANK_ENABLED,
        "rerank_model": RERANK_MODEL_NAME if RERANK_ENABLED else None,
        "rerank_device": RERANK_DEVICE if RERANK_ENABLED else None,
    }


@app.get("/text_embedding")
def get_text_embedding(text: str):
    try:
        logger.info("收到 text_embedding 请求，text_length=%s", len(text))
        vector = embeddings.embed_query(text)
        return {"embedding": vector}
    except Exception as exc:
        logger.exception("text_embedding 调用失败")
        return {"error": str(exc)}


@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        logger.info("收到 upload 请求，filename=%s", file.filename)
        file_content = await file.read()

        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_content.decode("gbk")
            except UnicodeDecodeError:
                return {"error": "文件解码失败！请确保上传的是纯文本文件 (如 .txt)。"}

        if not text.strip():
            return {"error": "上传的文件内容为空"}

        chunks = text_splitter.split_text(text)
        if not chunks:
            return {"error": "文本切块后无有效内容"}

        documents = [
            Document(
                page_content=chunk,
                metadata={"filename": file.filename, "chunk_index": index},
            )
            for index, chunk in enumerate(chunks)
        ]

        try:
            vector_db.add_documents(documents)
            logger.info("上传完成，filename=%s, chunks=%s", file.filename, len(chunks))
        except Exception as db_err:
            logger.exception("Chroma 入库失败，filename=%s", file.filename)
            return {"error": f"向量入库失败: {str(db_err)}"}

        return {"status": "success", "chunks_count": len(chunks)}

    except Exception as exc:
        logger.exception("upload 接口发生未知错误，filename=%s", file.filename)
        return {"error": f"接口发生未知错误: {str(exc)}"}


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        return {"error": "query 不能为空"}

    try:
        logger.info("收到 chat 请求，query=%s", query)

        initial_docs = vector_db.similarity_search(query, k=RETRIEVAL_TOP_K)
        logger.info("向量召回完成，retrieved=%s", len(initial_docs))

        if not initial_docs:
            logger.info("知识库无匹配内容，直接调用大模型。")
            response = chat_model.invoke(query)
            return {
                "answer": response.content,
                "source": "llm_fallback",
                "referenced_files": [],
                "retrieval_strategy": "llm_only",
            }

        top_n = min(RERANK_TOP_N, len(initial_docs))
        final_docs, scored_results, rerank_applied = rerank_documents(
            query=query,
            documents=initial_docs,
            top_n=top_n,
        )

        for index, item in enumerate(scored_results[:top_n], start=1):
            logger.info(
                "rerank top%s: score=%.4f, filename=%s",
                index,
                item["score"],
                item["filename"],
            )

        context = build_context(final_docs)
        source_files = unique_source_files(final_docs)

        prompt_template = """你是一个智能助手。请优先使用以下【参考资料】来回答用户的问题。
如果【参考资料】中没有包含能回答该问题的信息，请忽略参考资料，直接使用你自己的知识来回答。
请直接给出回答，不要在回答中说明“参考资料中没有提及”或“根据我的知识”。

【参考资料】：
{context}

【用户问题】：{query}
"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | chat_model | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query})

        logger.info(
            "chat 完成，rerank_applied=%s, referenced_files=%s",
            rerank_applied,
            source_files,
        )

        return {
            "answer": answer,
            "source": "rag_or_llm",
            "referenced_files": source_files,
            "retrieval_strategy": (
                "vector_search+rerank" if rerank_applied else "vector_search"
            ),
            "retrieved_chunks": len(initial_docs),
            "used_chunks": len(final_docs),
        }

    except Exception as exc:
        logger.exception("chat 接口发生错误，query=%s", query)
        return {"error": f"对话接口发生错误: {str(exc)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="info", access_log=True)
