import json
import time
from fastapi import FastAPI, HTTPException
import requests
from uuid import uuid4
from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, connections, utility
from pymilvus.client.types import LoadState

app = FastAPI(title="MacBook RAG Backend (Milvus + Ollama)")


MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "jie_knowledge_base"
DATA_DIR = "./docs"  # 存放你 Markdown 文件的文件夹
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3:567m"


class OllamaAPIEmbeddings(Embeddings):
    """Use Ollama's embed API directly to avoid httpx client 502s."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else ""
            raise RuntimeError(
                "Ollama embed request failed with status "
                f"{exc.response.status_code if exc.response else 'unknown'}: "
                f"{detail or exc}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.base_url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama embed API returned invalid JSON.") from exc

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama embed API returned no embeddings: {data}")

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


# 初始化模型 (确保你已经在终端执行了 ollama pull)
embeddings = OllamaAPIEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
llm = ChatOpenAI(
    model="gpt-5.4",
    api_key=SecretStr(
        "sk-846f1b178fbb2f6810f0cda48c69c9a7c2ffd29c8847887820bf7ee0a17a8945"
    ),
    base_url="https://gmncode.cn",
)


@app.get("/")
async def health_check():
    alias = f"healthcheck-{uuid4().hex}"
    try:
        connections.connect(alias=alias, uri=MILVUS_URI)
        collections = utility.list_collections(using=alias)
        return {
            "status": "ok",
            "message": "Milvus connected successfully.",
            "milvus_uri": MILVUS_URI,
            "collections": collections,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Milvus connection failed: {e}"
        ) from e
    finally:
        try:
            connections.disconnect(alias=alias)
        except Exception:
            pass


@app.post("/ingest", summary="Ingest data into the RAG system")
async def ingest_docs():
    """
    扫描 DATA_DIR 下的所有 Markdown 文件，切片并存入 Milvus
    """
    try:
        loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        docs = loader.load()

        # 输出当前的docs
        # print("docs: %s", docs)

        if not docs:
            return {"message": "No documents found in the directory."}

        # 文档切片
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
        )

        splits = text_splitter.split_documents(docs)

        texts = [doc.page_content for doc in splits]
        vectors = embeddings.embed_documents(texts)

        milvus_client = MilvusClient(uri=MILVUS_URI)

        if milvus_client.has_collection(COLLECTION_NAME):
            milvus_client.drop_collection(COLLECTION_NAME)

        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=len(vectors[0]),
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
            max_length=128,
            enable_dynamic_field=True,
        )

        rows = []
        for i, (doc, vector) in enumerate(zip(splits, vectors)):
            rows.append(
                {
                    "id": f"chunk-{i}",
                    "vector": vector,
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "chunk_index": i,
                }
            )

        milvus_client.insert(collection_name=COLLECTION_NAME, data=rows)
        milvus_client.load_collection(collection_name=COLLECTION_NAME)

        verification_client = MilvusClient(uri=MILVUS_URI)
        for _ in range(30):
            load_state = verification_client.get_load_state(COLLECTION_NAME)
            if load_state["state"] == LoadState.Loaded:
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f"Collection {COLLECTION_NAME} was not loaded in time."
            )

        time.sleep(2)

        return {"status": "success", "chunks_indexed": len(splits)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)
