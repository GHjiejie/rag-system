from fastapi import FastAPI, UploadFile
from langchain_ollama import OllamaEmbeddings
import uvicorn


app = FastAPI(title="RAG")

# 修正 1: 去掉 /api/embed 后缀，只保留基础地址
# 修正 2: 删掉模型名称末尾的空格 "bge-m3:567m " -> "bge-m3:567m"
embeddings = OllamaEmbeddings(
    model="bge-m3:567m",
    base_url="http://localhost:11434",
    client_kwargs={"trust_env": False},
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
    # 输出文件的内容
    print(file)
    # 获取文件的内容
    file_content = await file.read()
    return {"filename": file.filename, "content": file_content}


if __name__ == "__main__":
    # 建议 host 改为 0.0.0.0 以获得更好的兼容性
    uvicorn.run(app, host="0.0.0.0", port=9090)
