import argparse
import time
import uvicorn
import shutil
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.llm.gpt_client import GPTManager
from src.utils.processor import DocumentProcessor
from src.utils.token_counter import count_tokens
from src.utils.logger import get_logger

# 1. Configuration & CLI Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt-4o")
parser.add_argument("--chunk_size", type=int, default=1000)
parser.add_argument("--strategy", default="recursive")
args, _ = parser.parse_known_args()

app = FastAPI(title="Aion Industrial RAG API")
metrics = {"calls": 0, "history": []}
logger = get_logger(__name__)

# 2. Component Initialization
# Ensure directories exist before starting
os.makedirs("data/embeddings", exist_ok=True)
os.makedirs("data/outputs", exist_ok=True)

logger.info(f"Starting Aion RAG API — model={args.model}, chunk_size={args.chunk_size}, strategy={args.strategy}")

processor = DocumentProcessor(chunk_size=args.chunk_size)
gpt = GPTManager(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=args.model,
    persist_dir="./data/embeddings"
)


@app.post("/upload-pdfs", tags=["Ingestion"])
async def upload_multiple_pdfs(files: List[UploadFile] = File(...)):
    """Full Pipeline (Notebooks 01-03): Upload, Split, and Index documents."""
    results = []
    for file in files:
        logger.info(f"Uploading file: {file.filename}")                  # log upload start
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process: Load -> Split -> Embed
        docs = processor.load_documents(file_path)
        splits = processor.split_documents(docs, strategy=args.strategy)
        logger.info(f"Split {file.filename} into {len(splits)} chunks")  # log chunk count

        vectorstore = gpt.get_vectorstore()
        vectorstore.add_documents(splits)
        logger.info(f"Indexed {file.filename} into ChromaDB")            # log indexing done
        # FIX: Removed vectorstore.persist() — ChromaDB auto-persists
        # when persist_directory is set. Calling .persist() explicitly
        # was removed in chromadb >= 0.4.x and causes an AttributeError.

        results.append({"filename": file.filename, "chunks": len(splits)})
    return {"message": "Files indexed successfully", "details": results}


@app.post("/query", tags=["Chat"])
async def handle_query(user_id: str, user_input: str):
    """Execution (Notebooks 04-06): Retrieve and Answer with Chat History."""
    start_time = time.perf_counter()
    logger.info(f"Query from {user_id}: {user_input}")                   # log incoming query

    try:
        vectorstore = gpt.get_vectorstore()
        # Search k=3 results as used in Notebook 04
        chain = gpt.create_chat_chain(vectorstore.as_retriever(search_kwargs={"k": 3}))

        # Invoke the chain
        result = chain.invoke({"question": user_input})
        answer = result.get("answer", "No context found.")

        latency = time.perf_counter() - start_time
        tokens = count_tokens(answer, model=args.model)
        logger.info(f"Query answered — tokens={tokens}, latency={latency:.2f}s")  # log success

        # Logging for Dashboard
        metrics["calls"] += 1
        metrics["history"].append({
            "user": user_id,
            "latency": f"{latency:.2f}s",
            "tokens": tokens,
            "timestamp": time.time()
        })

        return {
            "answer": answer,
            "metrics": {"tokens": tokens, "latency": f"{latency:.2f}s"}
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")                          # log error
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Dashboard"])
async def get_metrics():
    """Provides usage statistics for visualization."""
    logger.info("Metrics endpoint called")                               # log metrics access
    return metrics


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)