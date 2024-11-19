from datetime import datetime, timedelta
from typing import Dict, Optional
from pydantic import BaseModel
import numpy as np

from pipeline.models import EmbeddedChunkedArticle

class FinancialContentMetadata(BaseModel):
    content_id: str
    timestamp: datetime
    content_type: str = "financial_news"
    title: str
    summary: Optional[str] = None

class FinancialFreshnessScorer:
    def __init__(self, 
                 max_age_hours: float = 24.0,  # Maximum age before severe decay
                 critical_age_hours: float = 4.0,  # Age at which decay begins
                 min_score: float = 0.1):  # Minimum score for very old content
        self.max_age_hours = max_age_hours
        self.critical_age_hours = critical_age_hours
        self.min_score = min_score
        
    def calculate_confidence(self, 
                           content_metadata: EmbeddedChunkedArticle,
                           current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate confidence score based on content age using exponential decay.
        Specifically designed for financial news where freshness is critical.
        """
        current_time = current_time or datetime.utcnow()
        
        # Calculate age in hours
        content_age = (current_time - content_metadata.updated_at).total_seconds() / 3600
        
        # Calculate confidence score
        if content_age <= self.critical_age_hours:
            # Content is very fresh - maximum confidence
            confidence = 1.0
        else:
            # Apply exponential decay after critical age
            decay_rate = 0.15  # Adjust this to control decay speed
            age_factor = content_age - self.critical_age_hours
            confidence = np.exp(-decay_rate * age_factor)
            
            # Ensure confidence doesn't go below minimum
            confidence = max(confidence, self.min_score)
        
        return {
            'content_id': content_metadata.article_id,
            'confidence_score': float(confidence),
            'age_hours': float(content_age),
            'timestamp': current_time.isoformat(),
            'is_fresh': content_age <= self.critical_age_hours,
            'freshness_category': self._get_freshness_category(content_age)
        }
    
    def _get_freshness_category(self, age_hours: float) -> str:
        """Categorize content based on age."""
        if age_hours <= 1:
            return "real-time"
        elif age_hours <= 4:
            return "very_fresh"
        elif age_hours <= 12:
            return "fresh"
        elif age_hours <= 24:
            return "recent"
        else:
            return "outdated"
    
    def get_boost_factor(self, confidence_score: float) -> float:
        """
        Calculate a boost factor for search rankings based on confidence score.
        This can be used to modify relevance scores in the RAG system.
        """
        return confidence_score ** 2  # Square the confidence to emphasize recency

def demonstrate_financial_scoring():
    scorer = FinancialFreshnessScorer()
    current_time = datetime.utcnow()
    
    # Create test cases with different ages
    test_cases = [
        FinancialContentMetadata(
            content_id="1",
            timestamp=current_time - timedelta(minutes=30),
            title="Breaking: Q3 Earnings Report"
        ),
        FinancialContentMetadata(
            content_id="2",
            timestamp=current_time - timedelta(hours=5),
            title="Market Analysis"
        ),
        FinancialContentMetadata(
            content_id="3",
            timestamp=current_time - timedelta(hours=20),
            title="Stock Recommendation"
        )
    ]
    
    results = []
    for case in test_cases:
        score = scorer.calculate_confidence(case)
        boost = scorer.get_boost_factor(score['confidence_score'])
        results.append({
            'title': case.title,
            'score': score,
            'boost_factor': boost
        })
    
    return results

class FinancialRAG:
    def __init__(self, freshness_scorer: FinancialFreshnessScorer):
        self.freshness_scorer = freshness_scorer
        self.documents = {}  # In practice, replace with your vector store
        
    def add_document(self, content: str, metadata: FinancialContentMetadata):
        """Add a document to the RAG system with freshness metadata."""
        self.documents[metadata.content_id] = {
            'content': content,
            'metadata': metadata
        }
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search documents with freshness-adjusted ranking.
        This is a simplified example - in practice, integrate with your vector store.
        """
        results = []
        current_time = datetime.utcnow()
        
        for doc_id, doc in self.documents.items():
            # Calculate base relevance (replace with actual vector similarity)
            base_relevance = 0.5  # Placeholder
            
            # Calculate freshness confidence
            freshness = self.freshness_scorer.calculate_confidence(
                doc['metadata'],
                current_time
            )
            
            # Calculate boost factor
            boost = self.freshness_scorer.get_boost_factor(
                freshness['confidence_score']
            )
            
            # Combine relevance and freshness
            final_score = base_relevance * boost
            
            results.append({
                'document': doc,
                'relevance': base_relevance,
                'freshness': freshness,
                'final_score': final_score
            })
        
        # Sort by final score and return top k
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]

# Example usage
if __name__ == "__main__":
    # Initialize the scorer
    scorer = FinancialFreshnessScorer(
        max_age_hours=24.0,    # Severe decay after 24 hours
        critical_age_hours=4.0, # Start decay after 4 hours
        min_score=0.1          # Minimum score for old content
    )
    
    # Create a RAG system
    rag = FinancialRAG(scorer)
    
    # Add some test documents
    current_time = datetime.utcnow()
    
    # Recent earnings report
    rag.add_document(
        "Company XYZ reported Q3 earnings above expectations...",
        FinancialContentMetadata(
            content_id="earnings_1",
            timestamp=current_time - timedelta(minutes=30),
            title="XYZ Q3 Earnings"
        )
    )
    
    # Older market analysis
    rag.add_document(
        "Market analysis shows increasing trend in tech sector...",
        FinancialContentMetadata(
            content_id="analysis_1",
            timestamp=current_time - timedelta(hours=6),
            title="Tech Sector Analysis"
        )
    )
    
    # Search with freshness-aware ranking
    results = rag.search("earnings", top_k=2)

    # Print results
    for result in results:
        print(result)