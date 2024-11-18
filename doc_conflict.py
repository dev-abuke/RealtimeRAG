from typing import List, Dict, Optional, Tuple
from datetime import datetime
import spacy
from transformers import pipeline
from pydantic import BaseModel
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ConflictType(Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    ENTITY = "entity"

@dataclass
class Conflict:
    type: ConflictType
    old_content: str
    new_content: str
    confidence: float
    entities_involved: List[str]
    resolution_strategy: str

class ConflictResolution(BaseModel):
    original_version: str
    new_version: str
    resolved_content: str
    confidence_score: float
    resolution_metadata: Dict

class ConflictHandler:
    def __init__(self):
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.zero_shot = pipeline("zero-shot-classification")
        self.ner = self.nlp.get_pipe("ner")
        
        # Define contradiction detection labels
        self.contradiction_labels = [
            "contradiction",
            "agreement",
            "neutral"
        ]

    def detect_conflicts(self, old_content: str, new_content: str) -> List[Conflict]:
        """Detect conflicts between old and new content versions."""
        conflicts = []
        
        # Extract entities and numbers
        old_doc = self.nlp(old_content)
        new_doc = self.nlp(new_content)
        
        # Check for factual contradictions
        factual_conflict = self._check_factual_conflicts(old_content, new_content)
        if factual_conflict:
            conflicts.append(factual_conflict)
        
        # Check for numerical conflicts
        numerical_conflicts = self._check_numerical_conflicts(old_doc, new_doc)
        conflicts.extend(numerical_conflicts)
        
        # Check for temporal conflicts
        temporal_conflicts = self._check_temporal_conflicts(old_doc, new_doc)
        conflicts.extend(temporal_conflicts)
        
        # Check for entity conflicts
        entity_conflicts = self._check_entity_conflicts(old_doc, new_doc)
        conflicts.extend(entity_conflicts)
        
        return conflicts

    def resolve_conflicts(self, conflicts: List[Conflict], 
                         resolution_strategy: str = "newest_wins") -> ConflictResolution:
        """Resolve detected conflicts using specified strategy."""
        resolved_content = ""
        confidence_score = 0.0
        resolution_metadata = {}
        
        for conflict in conflicts:
            if resolution_strategy == "newest_wins":
                resolved_content = conflict.new_content
                confidence_score = conflict.confidence
                resolution_metadata["strategy"] = "newest_wins"
                resolution_metadata["original_conflict_type"] = conflict.type.value
                
            elif resolution_strategy == "confidence_based":
                # Use content with higher confidence
                if conflict.confidence > 0.7:  # Confidence threshold
                    resolved_content = conflict.new_content
                    confidence_score = conflict.confidence
                else:
                    resolved_content = conflict.old_content
                    confidence_score = 1 - conflict.confidence
                    
                resolution_metadata["strategy"] = "confidence_based"
                resolution_metadata["confidence_threshold"] = 0.7
                
            elif resolution_strategy == "merge":
                resolved_content = self._merge_contents(
                    conflict.old_content,
                    conflict.new_content,
                    conflict.type
                )
                confidence_score = 0.5  # Merged content has medium confidence
                resolution_metadata["strategy"] = "merge"
                
        return ConflictResolution(
            original_version=conflicts[0].old_content,
            new_version=conflicts[0].new_content,
            resolved_content=resolved_content,
            confidence_score=confidence_score,
            resolution_metadata=resolution_metadata
        )

    def _check_factual_conflicts(self, old_content: str, new_content: str) -> Optional[Conflict]:
        """Check for factual contradictions using zero-shot classification."""
        result = self.zero_shot(
            sequences=[f"{old_content} [SEP] {new_content}"],
            candidate_labels=self.contradiction_labels,
            hypothesis_template="These statements {}"
        )
        
        if result['labels'][0] == "contradiction":
            return Conflict(
                type=ConflictType.FACTUAL,
                old_content=old_content,
                new_content=new_content,
                confidence=result['scores'][0],
                entities_involved=self._extract_entities(old_content + " " + new_content),
                resolution_strategy="newest_wins"
            )
        return None

    def _check_numerical_conflicts(self, old_doc, new_doc) -> List[Conflict]:
        """Detect conflicts in numerical values."""
        conflicts = []
        old_numbers = [(ent.text, float(ent.text)) for ent in old_doc.ents 
                      if ent.label_ == "CARDINAL" and ent.text.replace(".", "").isdigit()]
        new_numbers = [(ent.text, float(ent.text)) for ent in new_doc.ents 
                      if ent.label_ == "CARDINAL" and ent.text.replace(".", "").isdigit()]
        
        for old_num, old_val in old_numbers:
            for new_num, new_val in new_numbers:
                if abs(old_val - new_val) / max(old_val, new_val) > 0.1:  # 10% threshold
                    conflicts.append(Conflict(
                        type=ConflictType.NUMERICAL,
                        old_content=old_num,
                        new_content=new_num,
                        confidence=0.8,
                        entities_involved=[old_num, new_num],
                        resolution_strategy="newest_wins"
                    ))
        
        return conflicts

    def _check_temporal_conflicts(self, old_doc, new_doc) -> List[Conflict]:
        """Detect conflicts in dates and times."""
        conflicts = []
        old_dates = [ent for ent in old_doc.ents if ent.label_ == "DATE"]
        new_dates = [ent for ent in new_doc.ents if ent.label_ == "DATE"]
        
        for old_date in old_dates:
            for new_date in new_dates:
                if old_date.text != new_date.text:
                    conflicts.append(Conflict(
                        type=ConflictType.TEMPORAL,
                        old_content=old_date.text,
                        new_content=new_date.text,
                        confidence=0.9,
                        entities_involved=[old_date.text, new_date.text],
                        resolution_strategy="newest_wins"
                    ))
        
        return conflicts

    def _check_entity_conflicts(self, old_doc, new_doc) -> List[Conflict]:
        """Detect conflicts in named entities."""
        conflicts = []
        old_entities = {(ent.text, ent.label_) for ent in old_doc.ents}
        new_entities = {(ent.text, ent.label_) for ent in new_doc.ents}
        
        # Check for entity replacements
        for old_ent, old_label in old_entities:
            for new_ent, new_label in new_entities:
                if old_label == new_label and old_ent != new_ent:
                    conflicts.append(Conflict(
                        type=ConflictType.ENTITY,
                        old_content=old_ent,
                        new_content=new_ent,
                        confidence=0.7,
                        entities_involved=[old_ent, new_ent],
                        resolution_strategy="newest_wins"
                    ))
        
        return conflicts

    def _merge_contents(self, old_content: str, new_content: str, 
                       conflict_type: ConflictType) -> str:
        """Merge conflicting contents based on conflict type."""
        if conflict_type == ConflictType.NUMERICAL:
            return f"{new_content} (previously reported as {old_content})"
        elif conflict_type == ConflictType.TEMPORAL:
            return f"{new_content} (updated from {old_content})"
        elif conflict_type == ConflictType.ENTITY:
            return f"{new_content} (formerly {old_content})"
        else:
            return f"{new_content} [Note: This conflicts with earlier version: {old_content}]"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents]

# Example usage
def demonstrate_conflict_handling():
    handler = ConflictHandler()
    
    # Example 1: Numerical conflict
    old_version = "The company reported revenue of $10.5 million in Q2."
    new_version = "The company reported revenue of $12.8 million in Q2."
    
    conflicts = handler.detect_conflicts(old_version, new_version)
    resolution = handler.resolve_conflicts(conflicts, "merge")
    
    # Example 2: Entity conflict
    old_version = "Sarah Johnson was appointed as the new CEO."
    new_version = "Michael Chen was appointed as the new CEO."
    
    conflicts = handler.detect_conflicts(old_version, new_version)
    resolution = handler.resolve_conflicts(conflicts, "newest_wins")
    
    return conflicts, resolution

if __name__ == "__main__":
    conflicts, resolution = demonstrate_conflict_handling()