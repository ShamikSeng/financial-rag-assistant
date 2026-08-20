import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routes import router
from core.vector_database import initialize_empty_vectorstores
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
  logger.info("Starting up app...")
  initialize_empty_vectorstores()
  logger.info("Startup complete.")
  yield

app = FastAPI(title="RAG PDFBot", description="Chat with multiple PDFs :books:", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
  logger.info("Running app...")
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
