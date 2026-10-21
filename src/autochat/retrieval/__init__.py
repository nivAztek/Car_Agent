"""Hybrid retrieval over Cosmos DB (vector) + in-app BM25, with metadata filters resolved on `cars`."""

from autochat.retrieval.bm25 import ChunkStore, tokenize
from autochat.retrieval.catalog import CarCatalog, CatalogScope, CosmosCarCatalog, InMemoryCarCatalog
from autochat.retrieval.filters import SearchFilters, build_query, find_cars, resolve_car_ids
from autochat.retrieval.hybrid import HybridRetriever, RetrievedChunk, Retriever, build_retriever, rrf
from autochat.retrieval.resolver import MentionResolution, resolve_mentions

__all__ = [
    "CarCatalog",
    "CatalogScope",
    "ChunkStore",
    "CosmosCarCatalog",
    "HybridRetriever",
    "InMemoryCarCatalog",
    "MentionResolution",
    "RetrievedChunk",
    "Retriever",
    "SearchFilters",
    "build_query",
    "build_retriever",
    "find_cars",
    "resolve_car_ids",
    "resolve_mentions",
    "rrf",
    "tokenize",
]
