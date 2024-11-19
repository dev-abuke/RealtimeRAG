from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Set
from qdrant_client.http import models
from qdrant_client import QdrantClient
from loguru import logger
from datetime import datetime, timezone

from pipeline import constants
from pipeline.embeddings import EmbeddingModelSingleton, CrossEncoderModelSingleton
from pipeline.models import NewsArticle, EmbeddedChunkedArticle
from pipeline.qdrant import build_qdrant_client
from pipeline.models import ScoreCombiner, ArticleScores

@dataclass
class SearchResult:
    """Data class for search results."""
    articles: List[EmbeddedChunkedArticle]
    query: Optional[str] = None
    embedded_queries: Optional[List[List[float]]] = None

class QueryProcessor:
    """Handles query processing and embedding."""
    
    def __init__(self, embedding_model: EmbeddingModelSingleton):
        self.embedding_model = embedding_model
    
    def process_query(self, query: str) -> List[List[float]]:
        """
        Process and embed a query.
        
        Args:
            query: Raw query string
            
        Returns:
            List of embedded query chunks
        """
        cleaned_query = NewsArticle.clean_all(query)
        chunks = NewsArticle.compute_chunks(cleaned_query, self.embedding_model)
        return [self.embedding_model(chunk, to_list=True) for chunk in chunks]

class SearchRequestBuilder:
    """Builds search requests for Qdrant."""
    
    @staticmethod
    def build_requests(
        embedded_queries: List[List[float]],
        limit: int,
        include_payload: bool = True,
        include_vector: bool = True,
        filter_conditions: Optional[models.Filter] = None
    ) -> List[models.SearchRequest]:
        """
        Build search requests for each embedded query.
        
        Args:
            embedded_queries: List of query embeddings
            limit: Number of results to return
            include_payload: Whether to include payload in results
            include_vector: Whether to include vectors in results
            
        Returns:
            List of SearchRequest objects
        """
        search_requests = [
            models.SearchRequest(
                vector=embedded_query,
                limit=limit,
                with_payload=include_payload,
                with_vector=include_vector,
                filter=filter_conditions  # Apply filters here
            )
            for embedded_query in embedded_queries
        ]

        return search_requests

class ArticleProcessor:
    """Processes and ranks retrieved articles."""
    
    def __init__(self, cross_encoder_model: Optional[CrossEncoderModelSingleton] = None):
        self.cross_encoder_model = cross_encoder_model
    
    def process_retrieved_points(
        self,
        retrieved_points: List[List[models.ScoredPoint]],
    ) -> Set[EmbeddedChunkedArticle]:
        """
        Process retrieved points into articles.
        
        Args:
            retrieved_points: List of lists of scored points
            
        Returns:
            Set of unique EmbeddedChunkedArticle objects
        """
        articles = set()
        for chunk_points in retrieved_points:
            logger.debug(f"Processing {len(chunk_points)} retrieved points")
            chunk_articles = {
                EmbeddedChunkedArticle.from_retrieved_point(point)
                for point in chunk_points
            }
            articles.update(chunk_articles)
        return articles
    
    def rank_articles(
        self,
        articles: List[EmbeddedChunkedArticle],
        query: str,
        limit: int,
        use_time_decay: bool = False,
        scoring_method: str = "Weighted Average"
    ) -> List[EmbeddedChunkedArticle]:
        """
        Rank articles using cross-encoder or score sorting.
        
        Args:
            articles: List of articles to rank
            query: Original query string
            limit: Number of articles to return
            
        Returns:
            Ranked and limited list of articles
        """
        if self.cross_encoder_model:
            # TODO: Implement reranking
            articles = self.rerank(query, articles)
            # pass
        else:
            articles.sort(key=lambda x: x.score, reverse=True)

        if use_time_decay: # TODO this is for debug only change to use_time_decay
            articles = self._rank_using_time_decay(articles)
        
        articles = self._add_scores(articles)

        if scoring_method == "Weighted Average":
            articles.sort(key=lambda a: a.weighted_avg_score, reverse=True)
        elif scoring_method == "Harmonic Mean":
            articles.sort(key=lambda a: a.harmonic_mean_score, reverse=True)
        elif scoring_method == "Geometric Mean":
            articles.sort(key=lambda a: a.geometric_mean_score, reverse=True)
        else:
            raise ValueError(f"Unknown scoring method: {scoring_method}")

        article_slice = articles[:limit]
        
        return article_slice
    
    def _add_scores(self, articles: List[EmbeddedChunkedArticle]) -> List[EmbeddedChunkedArticle]:
        """
        Calculate and update scores for each article using various scoring methods.
        
        Args:
            articles: List of articles with initial scores.
            
        Returns:
            List of articles with updated scores including weighted average,
            harmonic mean, and geometric mean scores.
        """
        for article in articles:
            
            combiner = ScoreCombiner()

            scores = ArticleScores(
                similarity_score=article.score,
                cross_encoder_score=article.rerank_score,
                time_decay_score=article.decay_score
            )
        
            # Try different combination methods
            article.weighted_avg_score = combiner.weighted_average(scores)
            article.harmonic_mean_score = combiner.harmonic_mean(scores)
            article.geometric_mean_score = combiner.geometric_mean(scores)

            logger.info(f"The reranking scores {article.weighted_avg_score} : {article.harmonic_mean_score} : {article.geometric_mean_score}: similarity score: {article.score}, rerank crossencoder score: {article.rerank_score} time decay score: {article.decay_score} Updated at: {article.updated_at} Article TEXT: {article.text}")
        
        return articles
    
    def _rank_using_time_decay(self, articles: List[EmbeddedChunkedArticle]) -> List[EmbeddedChunkedArticle]:
        """
        Rank articles using time decay.

        This function takes a list of articles and updates their scores by applying a time decay
        function to the original scores. The time decay function is implemented using the
        FinancialFreshnessScorer class from the data_decay module. The scorer calculates a
        confidence score based on the age of the article, and this score is then used to
        adjust the original score.

        Args:
            articles (List[EmbeddedChunkedArticle]): List of articles to rank

        Returns:
            List[EmbeddedChunkedArticle]: List of articles with updated scores
        """
        from data_decay import FinancialFreshnessScorer

        # Initialize the freshness scorer
        freshness_scorer = FinancialFreshnessScorer()

        # Get the current time
        current_time = datetime.now(timezone.utc)

        # Iterate over each article
        for article in articles:
            # Calculate the freshness score
            freshness = freshness_scorer.calculate_confidence(
                article,
                current_time
            )

            # Log the freshness score
            logger.info(f"The freshnes score is: {freshness['confidence_score']}")

            # Update the article's score using the freshness score
            article.decay_score = freshness['confidence_score'] # article.score # * boost  # Adjust the score

        return articles
    
    def rerank(
        self, query: str, posts: list[EmbeddedChunkedArticle]
    ) -> list[EmbeddedChunkedArticle]:
        """
        Rerank articles using cross-encoder scores.

        Args:
            query: Original query string
            posts: List of articles to rerank

        Returns:
            Reranked list of articles with cross-encoder scores
        """
        import numpy as np

        logger.info("Reranking using Crossencoder...")

        # Create pairs of [query, post text]
        pairs = [[query, f"{post.text}"] for post in posts]

        # Compute cross-encoder scores for each pair
        cross_encoder_scores = self.cross_encoder_model(pairs)
        # Softmax Normalization - Option 1

        #     Converts scores to probabilities that sum to 1
        #     Preserves relative differences between scores
        #     Good when scores can be interpreted as logits (could be used if we toggle apply_softmax in crossencoder prediction)

        # Min-Max Normalization - Option 2

        #     Scales scores to [0,1] range linearly
        #     Preserves relative ordering
        #     Good for general score normalization

        # Normalize scores to [0,1] range using softmax 
        scores_array = np.array(cross_encoder_scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Subtract max for numerical stability
        normalized_scores = exp_scores / exp_scores.sum()
        
        # Normalize scores to [0,1] range using min-max normalization
        # scores_array = np.array(cross_encoder_scores)
        # normalized_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())

        # Sort the posts by cross-encoder scores in descending order
        ranked_posts = sorted(
            zip(posts, normalized_scores), key=lambda x: x[1], reverse=True
        )

        # Create a new list of reranked posts with cross-encoder scores
        reranked_posts = []
        for post, rerank_score in ranked_posts:
            post.rerank_score = rerank_score  # Set the rerank score on the post

            reranked_posts.append(post)

        return reranked_posts

class VectorDBRetriever:
    """Main retriever class for vector database search."""
    
    def __init__(
        self,
        cross_encoder_model: Optional[CrossEncoderModelSingleton] = CrossEncoderModelSingleton(),
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        """
        Initialize the retriever.
        
        Args:
            cross_encoder_model: Optional cross encoder for reranking
            collection_name: Name of the vector database collection
        """
        self._query_processor = QueryProcessor(EmbeddingModelSingleton())
        self._article_processor = ArticleProcessor(cross_encoder_model)
        self._vector_db_client: QdrantClient = build_qdrant_client()
        self._collection_name = collection_name
    
    def search(
        self,
        query: str,
        limit: int = 3,
        return_all: bool = False,
        filter_conditions: Optional[models.Filter] = None,
        use_time_decay: bool = False,
        scoring_method: str = "Weighted Average"
    ) -> List[EmbeddedChunkedArticle]:
        """
        Search for articles matching the query.
        
        Args:
            query: Search query string
            limit: Number of results to return
            return_all: Whether to return additional search metadata
            filter_conditions: Optional Qdrant filter conditions
            
        Returns:
            List of articles or SearchResult object with metadata
        """
        # Process and embed query
        embedded_queries = self._query_processor.process_query(query)
        
        # Adjust limit for reranking if needed
        search_limit = limit * 10 if self._article_processor.cross_encoder_model else limit
        
        # Build and execute search requests
        search_requests = SearchRequestBuilder.build_requests(
            embedded_queries=embedded_queries,
            limit=search_limit,
            filter_conditions=filter_conditions
        )
        
        retrieved_points = self._vector_db_client.search_batch(
            collection_name=self._collection_name,
            requests=search_requests,
        )
        logger.info(f"Retrieved {len(retrieved_points)} batches of points")
        
        # Process retrieved points into articles
        articles = self._article_processor.process_retrieved_points(retrieved_points)
        
        # Rank and limit articles
        ranked_articles = self._article_processor.rank_articles(
            articles=list(articles),
            query=query,
            limit=limit,
            use_time_decay=use_time_decay,
            scoring_method=scoring_method
        )
        
        # if return_all:
        #     return SearchResult(
        #         articles=ranked_articles,
        #         query=query,
        #         embedded_queries=embedded_queries
        #     )
        
        return ranked_articles
    def search_by_filters(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 3,
        return_all: bool = False,
        return_scores: bool = False,
        min_score: Optional[float] = None,
        use_time_decay: bool = False,
        scoring_method: str = "Weighted Average"
    ) -> List[EmbeddedChunkedArticle]:
        """
        Search with filters and optional reranking.
        
        Args:
            query: Search query
            symbols: List of stock symbols to filter
            date_from: Start date filter
            date_to: End date filter
            limit: Number of results to return
            return_all: Whether to return search metadata
            return_scores: Whether to return scoring details
            min_score: Minimum score threshold
            
        Returns:
            Filtered and ranked articles
        """
        # Build filter conditions
        must_conditions = []
        
        if symbols:
            must_conditions.append(
                models.FieldCondition(
                    key="symbols",
                    match=models.MatchAny(any=symbols)
                )
            )
            logger.debug(f"Added symbols filter: {symbols}")
        
        if date_from or date_to:
            range_condition = {}
            if date_from:
                range_condition["gte"] = date_from.isoformat()
            if date_to:
                range_condition["lte"] = date_to.isoformat()
                
            must_conditions.append(
                models.FieldCondition(
                    key="created_at",
                    range=models.Range(**range_condition)
                )
            )
            logger.debug(f"Added date range filter: {range_condition}")
        
        filter_conditions = models.Filter(must=must_conditions) if must_conditions else None
        
        try:
            # Update search requests with filter
            results = self.search(
                query=query,
                limit=limit,
                return_all=return_all,
                filter_conditions=filter_conditions,  # Pass filters to search
                use_time_decay=use_time_decay,
                scoring_method=scoring_method
            )
            
            # Apply minimum score filter if specified
            if min_score is not None and isinstance(results, list):
                results = [
                    article for article in results
                    if article.score >= min_score
                ]
             
            # Add scores to response if requested
            if return_scores and isinstance(results, list):
                for article in results:
                    pass
            
            return results
            
        finally:
            # Reset search parameters
            logger.debug("Resetting search parameters")
            # pass
            # self._vector_db_client.search_kwargs = {}


if __name__ == "__main__":
    # Initialize retriever
    retriever = VectorDBRetriever()
    
    # Simple search
    results = retriever.search("What happened with AAPL stock?")
    
    # Search with metadata
    detailed_results = retriever.search(
        "Latest news about Tesla",
        limit=5,
        return_all=True
    )