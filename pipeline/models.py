import hashlib, sys
from datetime import datetime
from typing import List, Optional, Tuple, Union

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
    headline: Optional[str]
    created_at: Optional[datetime]
    chunk_id: str
    full_raw_text: str
    text: str
    text_embedding: list
    score: Optional[float] = None
    rerank_score: Optional[float] = None

    @classmethod
    def from_retrieved_point(cls, point: Union[ScoredPoint, Record]) -> "EmbeddedChunkedArticle":
        logger.info(f"The type of point.payload['article_id']: {type(point.payload['article_id'])}")
        return cls(
            article_id=point.payload["article_id"],
            headline=point.payload["headline"],
            created_at=point.payload["created_at"],
            chunk_id=point.id,
            full_raw_text=point.payload["full_raw_text"],
            text=point.payload["text"],
            text_embedding=point.vector,
            score=point.score if hasattr(point, "score") else None
        )
    
    def __str__(self) -> str:
        return f"EmbeddedChunkedPost(post_id={self.article_id}, chunk_id={self.chunk_id}, has_image={bool(False)}, text_embedding_length={len(self.text_embedding)})"

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

    @staticmethod
    def clean_all(text: str) -> str:
        cleaned_text = unbold_text(text)
        article_elements = partition_html(text=cleaned_text)
        print(f"Number of article elements: {len(article_elements)}")
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

        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self
