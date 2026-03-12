"""Configuration settings loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
DOCS_DIR: str = os.getenv("DOCS_DIR", "./docs")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))
