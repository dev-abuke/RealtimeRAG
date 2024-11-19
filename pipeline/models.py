import hashlib
from datetime import datetime
from typing import List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.partition.html import partition_html
from unstructured.staging.huggingface import chunk_by_attention_window
from qdrant_client.models import ScoredPoint, Record
from loguru import logger

from pipeline.embeddings import EmbeddingModelSingleton
from pipeline.utils import unbold_text, unitalic_text, replace_urls_with_placeholder, remove_emojis_and_symbols

# logger.add(sys.stderr, format="{time} {level} {message} {line}", filter="sec_filing", level="INFO")

class EmbeddedChunkedArticle(BaseModel):
    article_id: Union[int, str]
    symbol: Optional[List[str]]
    url: Optional[str]
    author: Optional[str]
    headline: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    chunk_id: str
    full_raw_text: str
    text: str
    text_embedding: list
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    decay_score: Optional[float] = None
    geometric_mean_score: Optional[float] = None
    harmonic_mean_score: Optional[float] = None
    weighted_avg_score: Optional[float] = None
    fusion_score: Optional[float] = None

    @classmethod
    def from_retrieved_point(cls, point: Union[ScoredPoint, Record]) -> "EmbeddedChunkedArticle":
        logger.info(f"The type of point.payload['article_id']: {point.payload['article_id']}")
        return cls(
            article_id=point.payload["article_id"],
            symbol=point.payload["symbols"],
            url=point.payload["url"],
            author=point.payload["author"],
            updated_at=point.payload["updated_at"],
            headline=point.payload["headline"],
            created_at=point.payload["created_at"],
            chunk_id=point.id,
            full_raw_text=point.payload["full_raw_text"],
            text=point.payload["text"],
            text_embedding=point.vector,
            score=point.score if hasattr(point, "score") else None
        )
    
    def __str__(self) -> str:
        return f"EmbeddedChunkedPost(post_id={self.article_id}, chunk_id={self.chunk_id}, text_embedding_length={len(self.text_embedding)})"

    def __hash__(self) -> int:
        return hash(self.chunk_id)

class NewsArticle(BaseModel):
    """
    Represents a news article.

    Attributes:
        id (int): News article ID
        headline (str): Headline or title of the article
        summary (str): Summary text for the article (may be first sentence of content)
        author (str): Original author of news article
        created_at (datetime): Date article was created (RFC 3339)
        updated_at (datetime): Date article was updated (RFC 3339)
        url (Optional[str]): URL of article (if applicable)
        content (str): Content of the news article (might contain HTML)
        symbols (List[str]): List of related or mentioned symbols
        source (str): Source where the news originated from (e.g. Benzinga)
    """

    id: int
    headline: str
    summary: str
    author: str
    created_at: datetime
    updated_at: datetime
    url: Optional[str]
    content: str
    symbols: List[str]
    source: str
    images: Optional[List[Union[Optional[str], Optional[dict]]]] = None

    @staticmethod
    def clean_all(text: str) -> str:
        cleaned_text = unbold_text(text)
        article_elements = partition_html(text=cleaned_text)
        logger.info(f"Number of article elements: {len(article_elements)}")
        joined_article = " ".join([str(x) for x in article_elements])
        cleaned_text = unitalic_text(joined_article)
        cleaned_text = remove_emojis_and_symbols(cleaned_text)
        cleaned_text = clean(cleaned_text)
        cleaned_text = replace_unicode_quotes(cleaned_text)
        cleaned_text = clean_non_ascii_chars(cleaned_text)
        cleaned_text = replace_urls_with_placeholder(cleaned_text)
        
        return cleaned_text

    @staticmethod
    def compute_chunks(text: str, model: EmbeddingModelSingleton = EmbeddingModelSingleton()) -> List[str]:
        """
        Computes the chunks of the document.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the chunks.

        Returns:
            Document: The document object with the computed chunks.
        """

        chunked_item = chunk_by_attention_window(text, model.tokenizer, max_input_size=model.max_input_length)

        return chunked_item
    
    def to_document(self) -> "Document":
        """
        Converts the news article to a Document object.

        Returns:
            Document: A Document object representing the news article.
        """

        document_id = hashlib.md5(self.content.encode()).hexdigest()
        logger.info(f"Document ID: {document_id}")

        document = Document(id=document_id)

        cleaned_content = self.clean_all(self.content)

        cleaned_headline = self.clean_all(self.headline)

        cleaned_summary = self.clean_all(self.summary)

        document.text = [cleaned_headline, cleaned_summary, cleaned_content]

        document.metadata["headline"] = cleaned_headline
        document.metadata["summary"] = cleaned_summary
        document.metadata["url"] = self.url
        document.metadata["symbols"] = self.symbols
        document.metadata["author"] = self.author
        document.metadata["created_at"] = self.created_at
        document.metadata["updated_at"] = self.updated_at
        document.metadata["source"] = self.source
        document.metadata["full_raw_text"] = self.content
        document.metadata["article_id"] = self.id

        return document


class Document(BaseModel):
    """
    A Pydantic model representing a document.

    Attributes:
        id (str): The ID of the document.
        group_key (Optional[str]): The group key of the document.
        metadata (dict): The metadata of the document.
        text (list): The text of the document.
        chunks (list): The chunks of the document.
        embeddings (list): The embeddings of the document.

    Methods:
        to_payloads: Returns the payloads of the document.
        compute_chunks: Computes the chunks of the document.
        compute_embeddings: Computes the embeddings of the document.
    """

    id: str
    group_key: Optional[str] = None
    metadata: dict = {}
    text: list = []
    chunks: list = []
    embeddings: list = []

    def to_payloads(self) -> Tuple[List[str], List[dict]]:
        from copy import deepcopy 
        """
        Returns the payloads of the document.

        Returns:
            Tuple[List[str], List[dict]]: A tuple containing the IDs and payloads of the document.
        """

        processed = [(
            hashlib.md5(chunk.encode()).hexdigest(),
            {**deepcopy(self.metadata), "text": chunk}
        ) for chunk in self.chunks]
        
        # Unzip the results
        ids, payloads = zip(*processed) if processed else ([], [])
        
        # Log verification
        logger.info(f"Total chunks: {len(self.chunks)}")
        logger.info(f"Unique chunk IDs: {len(set(ids))}")
        logger.info(f"Total payloads: {len(payloads)}")
        
        return list(ids), list(payloads)

    def compute_chunks(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Computes the chunks of the document.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the chunks.

        Returns:
            Document: The document object with the computed chunks.
        """

        logger.info(f"Computing chunks for document {self.id} and headline {self.metadata['headline']}...")

        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)

        return self

    def compute_embeddings(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Computes the embeddings for each chunk in the document using the specified embedding model.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the embeddings.

        Returns:
            Document: The document object with the computed embeddings.
        """

        logger.info(f"Computing embeddings for document {self.id} and headline {self.metadata['headline']}...")

        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self

@dataclass
class ArticleScores:
    """Container for article scores."""
    similarity_score: float
    cross_encoder_score: float
    time_decay_score: float
    
    def __post_init__(self):
        """Validate scores are between 0 and 1."""
        for field, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be between 0 and 1")

class ScoreCombiner:
    """Combines multiple relevance scores using different strategies."""
    
    def __init__(
        self,
        similarity_weight: float = 0.15,  # Default weight for similarity score
        cross_encoder_weight: float = 0.35,  # Default weight for cross-encoder score
        time_decay_weight: float = 0.5,  # Default weight for time decay score
        use_dynamic_weights: bool = False  # Flag to indicate dynamic weight usage
    ):
        """
        Initialize the score combiner with specified weights and options.
        
        Args:
            similarity_weight (float): The weight assigned to the similarity score.
            cross_encoder_weight (float): The weight assigned to the cross-encoder score.
            time_decay_weight (float): The weight assigned to the time decay score.
            use_dynamic_weights (bool): Whether to enable dynamic weight adjustment.
        """
        # Store static weights in a dictionary for easy access
        self.static_weights = {
            'similarity': similarity_weight,  # Assign similarity weight
            'cross_encoder': cross_encoder_weight,  # Assign cross-encoder weight
            'time_decay': time_decay_weight  # Assign time decay weight
        }
        
        # Set the flag indicating whether dynamic weights should be used
        self.use_dynamic_weights = use_dynamic_weights
        
        # Initialize a MinMaxScaler instance for potential score normalization
        self.scaler = MinMaxScaler()
    
    def weighted_average(self, scores: ArticleScores) -> float:
        """
        Combine scores using weighted average.

        This method takes an ArticleScores object and returns a single
        combined score. The combined score is calculated by multiplying
        each score by its corresponding weight and summing the results.

        The weights are determined by the _get_weights() method, which can
        return either static weights or dynamic weights based on the
        score distributions.

        Args:
            scores (ArticleScores): The scores to combine

        Returns:
            float: The combined score
        """
        weights = self._get_weights(scores)
        
        # Calculate the combined score by multiplying each score by its
        # corresponding weight and summing the results
        combined_score = (
            weights['similarity'] * scores.similarity_score +
            weights['cross_encoder'] * scores.cross_encoder_score +
            weights['time_decay'] * scores.time_decay_score
        )
        
        return combined_score

    
    def _get_weights(self, scores: ArticleScores) -> dict:
        """
        Get weights for score combination.

        When self.use_dynamic_weights is True, this method dynamically adjusts
        the weights based on the variance of the input scores. The idea is that
        if the scores are very similar (low variance), we should assign equal
        weights to each score. If the variance is high, it means the scores are
        quite different, so we should favor the cross-encoder score.

        Otherwise, this method simply returns the static weights stored in
        self.static_weights.
        """
        if not self.use_dynamic_weights:
            return self.static_weights
        
        # Calculate the variance of the scores
        score_variance = np.var([
            scores.similarity_score,
            scores.cross_encoder_score,
            scores.time_decay_score
        ])
        
        # Adjust weights based on variance
        if score_variance < 0.1:  # Low variance, scores are similar
            # Assign equal weights to each score
            return {
                'similarity': 0.33,
                'cross_encoder': 0.34,
                'time_decay': 0.33
            }
        else:  # High variance, favor cross-encoder
            # Favor the cross-encoder score by assigning it a higher weight
            return {
                'similarity': 0.25,
                'cross_encoder': 0.5,
                'time_decay': 0.25
            }
    
    def harmonic_mean(self, scores: ArticleScores) -> float:
        """
        Combine scores using weighted harmonic mean.

        The harmonic mean is a way to combine multiple scores into a single
        score. It's similar to the arithmetic mean, but it's more sensitive to
        low scores. The harmonic mean is calculated as:

            sum(weights) / sum(weights / scores)

        where weights are the weights for each score, and scores are the
        individual scores.

        In this implementation, we add a small value (1e-10) to each score
        to avoid division by zero. This is because the harmonic mean is
        undefined if any of the scores are zero.
        """
        weights = self._get_weights(scores)
        
        # Calculate the denominator of the harmonic mean
        denominator = sum(
            weight / (score + 1e-10)  # Avoid division by zero
            for weight, score in zip(
                weights.values(),
                [scores.similarity_score, scores.cross_encoder_score, scores.time_decay_score]
            )
        )
        
        # Calculate the harmonic mean
        return sum(weights.values()) / denominator
    
    def geometric_mean(self, scores: ArticleScores) -> float:
        """Combine scores using weighted geometric mean."""
        
        # Retrieve the weights for each score type
        weights = self._get_weights(scores)
        
        # Calculate the weighted geometric mean
        # The geometric mean is calculated by raising each score to the power of its weight,
        # multiplying all the resulting values together, and taking the nth root,
        # where n is the number of scores. In this implementation, we directly use
        # the power of the product to achieve this.
        weighted_geometric_mean = np.prod([
            score ** weight  # Raise each score to the power of its weight
            for score, weight in zip(
                [scores.similarity_score, scores.cross_encoder_score, scores.time_decay_score],
                weights.values()
            )
        ])
        
        return weighted_geometric_mean
    
    def rank_fusion(self, articles_scores: List[ArticleScores]) -> List[float]:
        """
        Combine scores using rank fusion.

        Rank fusion is a method for combining scores from different sources
        by ranking each score type separately and then combining the ranks
        using weights.
        """
        # Get rankings for each score type
        # We use the _get_ranks method to convert the scores to ranks
        # This is done by sorting the scores in descending order and then
        # taking the index of each score in the sorted list as its rank
        # The rank is then normalized by dividing by the total number of scores
        similarity_ranks = self._get_ranks([s.similarity_score for s in articles_scores])
        cross_encoder_ranks = self._get_ranks([s.cross_encoder_score for s in articles_scores])
        time_decay_ranks = self._get_ranks([s.time_decay_score for s in articles_scores])
        
        # Combine ranks using weights
        # We use the static_weights attribute to get the weights for each score type
        # The weights are used to calculate a weighted average of the ranks
        weights = self.static_weights
        combined_ranks = []
        
        for i in range(len(articles_scores)):
            # Calculate the weighted rank for each article
            # The weighted rank is a linear combination of the ranks for each score type
            # The weights are used to control the relative importance of each score type
            weighted_rank = (
                weights['similarity'] * similarity_ranks[i] +
                weights['cross_encoder'] * cross_encoder_ranks[i] +
                weights['time_decay'] * time_decay_ranks[i]
            )
            # Append the weighted rank to the list of combined ranks
            combined_ranks.append(weighted_rank)
        
        # Normalize the combined ranks to [0,1]
        # We use the _normalize_scores method to normalize the combined ranks
        # This is done by subtracting the minimum value and dividing by the range
        return self._normalize_scores(combined_ranks)
    
    @staticmethod
    def _get_ranks(scores: List[float]) -> List[float]:
        """Convert scores to ranks."""
        return [
            sorted(scores, reverse=True).index(score) / len(scores)
            for score in scores
        ]
    
    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """
        Normalize scores to [0,1] range.

        This is done by subtracting the minimum value and dividing by the range.
        """
        # Find the minimum and maximum scores
        # This is done to calculate the range of the scores
        min_score = min(scores)
        max_score = max(scores)

        # Normalize the scores
        # This is done by subtracting the minimum value and dividing by the range
        return [
            # For each score, subtract the minimum value
            # This will shift the scores so that the minimum value is 0
            (score - min_score)
            # Then divide by the range
            # This will scale the scores so that the maximum value is 1
            / (max_score - min_score)
            for score in scores
        ]
