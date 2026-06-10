"""
RAG over research PDFs using LlamaIndex + Supabase pgvector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import (  # type: ignore[import-not-found]
    Document,
    Settings as LISettings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore[import-not-found]
from llama_index.vector_stores.supabase import SupabaseVectorStore  # type: ignore[import-not-found]
from pypdf import PdfReader

from glitz_quant.settings import get_app_config, get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


def _configure_llama_index() -> None:
    s = get_settings()
    cfg = get_app_config().get("rag", {})
    if s.openai_api_key:
        LISettings.embed_model = OpenAIEmbedding(
            model=cfg.get("embedding_model", "text-embedding-3-large"),
            api_key=s.openai_api_key.get_secret_value(),
        )
    LISettings.chunk_size = int(cfg.get("chunk_size", 1024))
    LISettings.chunk_overlap = int(cfg.get("chunk_overlap", 128))


class RAGStore:
    """Wraps a Supabase pgvector-backed VectorStoreIndex."""

    def __init__(self) -> None:
        _configure_llama_index()
        s = get_settings()
        cfg = get_app_config().get("rag", {})
        if not s.supabase_db_url:
            raise RuntimeError("SUPABASE_DB_URL required for RAG store")
        self.collection = cfg.get("collection", "research_documents")
        self.vector_store = SupabaseVectorStore(
            postgres_connection_string=s.supabase_db_url.get_secret_value(),
            collection_name=self.collection,
            dimension=3072,
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index: VectorStoreIndex | None = None

    def open(self) -> None:
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
        )
        log.info("rag_index_opened", collection=self.collection)

    def ingest_pdf(self, path: str | Path, source_label: str | None = None) -> int:
        """Read a PDF, chunk it, embed, store. Returns chunk count."""
        path = Path(path)
        reader = PdfReader(str(path))
        docs: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source": source_label or str(path.name),
                        "page": i + 1,
                    },
                )
            )
        if not docs:
            return 0
        idx = VectorStoreIndex.from_documents(
            documents=docs,
            storage_context=self.storage_context,
        )
        self.index = idx
        log.info("pdf_ingested", path=str(path), chunks=len(docs))
        return len(docs)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.index is None:
            self.open()
        k = top_k or int(get_app_config().get("rag", {}).get("top_k", 8))
        retriever = self.index.as_retriever(similarity_top_k=k)  # type: ignore[union-attr]
        nodes = retriever.retrieve(query)
        return [
            {
                "score": float(getattr(n, "score", 0.0) or 0.0),
                "text": n.get_content(),
                "metadata": dict(n.node.metadata),
            }
            for n in nodes
        ]
