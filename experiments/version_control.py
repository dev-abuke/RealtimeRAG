from datetime import datetime
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel
import numpy as np
from collections import defaultdict

class DocumentVersion(BaseModel):
    version_id: str
    content: str
    embedding: Optional[List[float]]
    timestamp: datetime
    is_active: bool
    metadata: Dict = {}
    
class Document(BaseModel):
    doc_id: str
    current_version_id: str
    versions: Dict[str, DocumentVersion]
    
class VersionControlledRAG:
    def __init__(self, embedding_dimension: int = 384):
        self.documents: Dict[str, Document] = {}
        self.embedding_dimension = embedding_dimension
        self.version_index = defaultdict(list)  # Maps timestamps to doc versions
        
    def add_document(self, content: str, metadata: Dict = {}) -> str:
        """Add a new document to the system."""
        doc_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        
        # Create initial version
        version = DocumentVersion(
            version_id=version_id,
            content=content,
            embedding=self._generate_embedding(content),
            timestamp=datetime.utcnow(),
            is_active=True,
            metadata=metadata
        )
        
        # Create document with initial version
        document = Document(
            doc_id=doc_id,
            current_version_id=version_id,
            versions={version_id: version}
        )
        
        self.documents[doc_id] = document
        self._update_version_index(doc_id, version)
        return doc_id
    
    def update_document(self, doc_id: str, new_content: str, metadata: Dict = {}) -> str:
        """Create a new version of an existing document."""
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
            
        # Create new version
        version_id = str(uuid.uuid4())
        new_version = DocumentVersion(
            version_id=version_id,
            content=new_content,
            embedding=self._generate_embedding(new_content),
            timestamp=datetime.utcnow(),
            is_active=True,
            metadata=metadata
        )
        
        # Deactivate previous version
        current_version = self.documents[doc_id].versions[self.documents[doc_id].current_version_id]
        current_version.is_active = False
        
        # Update document
        self.documents[doc_id].versions[version_id] = new_version
        self.documents[doc_id].current_version_id = version_id
        
        self._update_version_index(doc_id, new_version)
        return version_id
    
    def get_document_at_time(self, doc_id: str, timestamp: datetime) -> Optional[DocumentVersion]:
        """Retrieve the version of a document that was active at a specific time."""
        if doc_id not in self.documents:
            return None
            
        # Get all versions of the document
        versions = self.documents[doc_id].versions
        
        # Find the most recent version before or at the specified timestamp
        valid_versions = [v for v in versions.values() if v.timestamp <= timestamp]
        if not valid_versions:
            return None
            
        return max(valid_versions, key=lambda x: x.timestamp)
    
    def search_version_history(
        self,
        query_embedding: List[float],
        timestamp: Optional[datetime] = None,
        top_k: int = 5
    ) -> List[tuple[str, DocumentVersion, float]]:
        """Search for similar documents at a specific point in time."""
        results = []
        
        # If no timestamp provided, use current time
        search_time = timestamp or datetime.utcnow()
        
        # Search through all documents
        for doc_id, document in self.documents.items():
            version = self.get_document_at_time(doc_id, search_time)
            if version and version.embedding:
                similarity = self._calculate_similarity(query_embedding, version.embedding)
                results.append((doc_id, version, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def _generate_embedding(self, content: str) -> List[float]:
        """Placeholder for embedding generation - replace with actual embedding model."""
        # Simulate embedding generation with random vector
        return list(np.random.rand(self.embedding_dimension))
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _update_version_index(self, doc_id: str, version: DocumentVersion):
        """Update the temporal index with the new version."""
        self.version_index[version.timestamp].append((doc_id, version.version_id))

# Example usage
def demonstrate_version_control():
    rag = VersionControlledRAG()
    
    # Add initial document
    doc_id = rag.add_document(
        content="Initial version of the document",
        metadata={"source": "user1", "category": "news"}
    )
    
    # Update document
    rag.update_document(
        doc_id=doc_id,
        new_content="Updated version of the document",
        metadata={"source": "user1", "category": "news", "update_reason": "content correction"}
    )
    
    # Get document at specific time
    past_time = datetime.utcnow()
    version = rag.get_document_at_time(doc_id, past_time)
    
    return rag, doc_id, version

if __name__ == "__main__":
    rag, doc_id, version = demonstrate_version_control()