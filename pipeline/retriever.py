from typing import Optional, Union
from qdrant_client.http import models
from loguru import logger

from pipeline import constants  
from pipeline.embeddings import EmbeddingModelSingleton, CrossEncoderModelSingleton
from pipeline.models import NewsArticle, EmbeddedChunkedArticle
from pipeline.qdrant import build_qdrant_client


class QdrantVectorDBRetriever:
    def __init__(
        self,
        cross_encoder_model: Optional[CrossEncoderModelSingleton] = CrossEncoderModelSingleton(),
    ):
        self._embedding_model = EmbeddingModelSingleton()
        self._vector_db_client = build_qdrant_client()
        self._cross_encoder_model = cross_encoder_model
        self._vector_db_collection = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME

    def search(
        self, query: str, limit: int = 3, return_all: bool = False
    ) -> Union[list[EmbeddedChunkedArticle], dict[str, list]]:
        embdedded_queries = self.embed_query(query)

        if self._cross_encoder_model:
            original_limit = limit
            limit = limit * 10
        else:
            original_limit = limit

        search_queries = [
            models.SearchRequest(
                vector=embedded_query, limit=limit, with_payload=True, with_vector=True
            )
            for embedded_query in embdedded_queries
        ]
        retrieved_points = self._vector_db_client.search_batch(
            collection_name=self._vector_db_collection,
            requests=search_queries,
        )
        logger.info(f"The Length of retrieved_points: {len(retrieved_points)}")

        articles = set()
        for chunk_retrieved_points in retrieved_points:
            logger.info(f"The Length of chunk_retrieved_points: {len(chunk_retrieved_points)}, and chunk retrieved points: {chunk_retrieved_points}") 
            logger.info(f"Type of chunk_retrieved_points: {type(chunk_retrieved_points[0].id)}")
            articles.update(
                {
                    EmbeddedChunkedArticle.from_retrieved_point(point)
                    for point in chunk_retrieved_points
                }
            )
        articles = list(articles)

        if self._cross_encoder_model:
            # TODO: Rerank implementation here
            pass
            # articles = self.rerank(query, articles)
        else:
            articles = sorted(articles, key=lambda x: x.score, reverse=True)

        articles = articles[:original_limit]

        if return_all:
            return {
                "articles": articles,
                "query": query,
                "embdedded_queries": embdedded_queries,
            }

        return articles

    def embed_query(self, query: str) -> list[list[float]]:
        cleaned_query = NewsArticle.clean_all(query)
        chunks = NewsArticle.compute_chunks(cleaned_query, self._embedding_model)
        embdedded_queries = [
            self._embedding_model(chunk, to_list=True) for chunk in chunks
        ]

        return embdedded_queries