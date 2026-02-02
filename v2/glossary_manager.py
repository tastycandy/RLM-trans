"""
RLM Glossary Manager
Manages glossary entries with conflict resolution algorithm
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from rlm_state import TermEntry, PresetType


class ConflictResolutionRule(str, Enum):
    """Conflict resolution priority rules"""
    PRESET_FIRST = "preset_first"
    DOCUMENT_INITIAL = "document_initial"
    MAJORITY = "majority"
    MOST_RECENT = "most_recent"


@dataclass
class GlossaryConflict:
    """Represents a glossary conflict"""
    term: str
    options: List[str]
    sources: List[str]  # Where each option came from
    rule_applied: ConflictResolutionRule


class GlossaryManager:
    """
    Manages glossary entries with deterministic conflict resolution.
    """

    def __init__(self, conflict_rule: ConflictResolutionRule = ConflictResolutionRule.MAJORITY):
        """
        Initialize glossary manager.

        Args:
            conflict_rule: Rule for resolving conflicts
        """
        self.conflict_rule = conflict_rule
        self._conflict_history: List[GlossaryConflict] = []

    def add_term(self, source: str, target: str, confidence: float = 0.7,
                 source_chunks: Optional[List[int]] = None,
                 is_hard: bool = False,
                 preset_source: Optional[str] = None):
        """
        Add or update a glossary term.

        Args:
            source: Source term
            target: Target translation
            confidence: Confidence score (0.0-1.0)
            source_chunks: Chunk indices where term appears
            is_hard: Whether this is a hard glossary term
            preset_source: Which preset provided this term (if applicable)

        Returns:
            True if term was added, False if conflict occurred
        """
        # Check for conflicts
        existing = self._find_existing_term(source)

        if existing:
            return self._resolve_conflict(source, existing, target, confidence, source_chunks, is_hard, preset_source)

        # Add new term
        term = TermEntry(
            source=source,
            target=target,
            confidence=confidence,
            source_chunk_indices=source_chunks or [],
            is_hard=is_hard
        )

        return True

    def resolve_all_conflicts(self, glossary: Dict[str, str]) -> Dict[str, str]:
        """
        Resolve all conflicts in glossary dictionary.

        Args:
            glossary: Glossary dict to resolve

        Returns:
            Resolved glossary dict
        """
        conflicts = self._find_all_conflicts(glossary)

        resolved = glossary.copy()

        for conflict in conflicts:
            # Apply conflict resolution rule
            resolved_term = self._apply_conflict_resolution(conflict)

            resolved[conflict.term] = resolved_term

        return resolved

    def get_conflicts(self) -> List[GlossaryConflict]:
        """Get list of all conflicts in glossary"""
        return self._conflict_history

    def clear_conflicts(self):
        """Clear conflict history"""
        self._conflict_history.clear()

    def _find_existing_term(self, source: str) -> Optional[TermEntry]:
        """
        Find existing term entry.

        Args:
            source: Source term to find

        Returns:
            Existing term entry or None
        """
        # Check all presets first, then document-specific
        # In real implementation, would check preset-specific glossaries

        return None

    def _find_all_conflicts(self, glossary: Dict[str, str]) -> List[GlossaryConflict]:
        """
        Find all conflicts in glossary dictionary.

        Args:
            glossary: Glossary dict to check

        Returns:
            List of conflicts
        """
        # Count occurrences of each source term
        term_counts: Dict[str, Dict[str, int]] = {}

        for source, target in glossary.items():
            if source not in term_counts:
                term_counts[source] = {}
            term_counts[source][target] = term_counts[source].get(target, 0) + 1

        # Find conflicts (terms with multiple targets)
        conflicts = []

        for source, targets in term_counts.items():
            if len(targets) > 1:
                conflicts.append(GlossaryConflict(
                    term=source,
                    options=list(targets.keys()),
                    sources=[f"target_{k}" for k in targets.keys()],  # Simplified
                    rule_applied=ConflictResolutionRule.MAJORITY
                ))

        return conflicts

    def _resolve_conflict(self, source: str, existing: TermEntry, new_target: str,
                         new_confidence: float, new_chunks: Optional[List[int]],
                         new_is_hard: bool, preset_source: Optional[str]) -> bool:
        """
        Resolve conflict between existing and new term.

        Args:
            source: Source term
            existing: Existing term entry
            new_target: New target translation
            new_confidence: New confidence score
            new_chunks: New chunk indices
            new_is_hard: New hard flag
            preset_source: Preset source

        Returns:
            True if updated, False if kept existing
        """
        # Apply conflict resolution rule
        decision = self._apply_conflict_resolution_decision(
            source, existing, new_target, new_confidence, preset_source
        )

        if decision == "keep_existing":
            # Record conflict in history
            self._conflict_history.append(GlossaryConflict(
                term=source,
                options=[existing.target, new_target],
                sources=["existing", "new"],
                rule_applied=self.conflict_rule
            ))
            return False
        else:
            # Update existing term
            existing.target = new_target
            existing.confidence = max(existing.confidence, new_confidence)

            if new_chunks:
                existing.source_chunk_indices.extend(new_chunks)

            existing.usage_count += 1

            # Record conflict in history
            self._conflict_history.append(GlossaryConflict(
                term=source,
                options=[existing.target, new_target],
                sources=["updated", "new"],
                rule_applied=self.conflict_rule
            ))

            return True

    def _apply_conflict_resolution_decision(self, source: str, existing: TermEntry,
                                            new_target: str, new_confidence: float,
                                            preset_source: Optional[str]) -> str:
        """
        Apply conflict resolution rule to decide which term to keep.

        Args:
            source: Source term
            existing: Existing term entry
            new_target: New target translation
            new_confidence: New confidence score
            preset_source: Preset source

        Returns:
            "keep_existing" or "update_existing"
        """
        if self.conflict_rule == ConflictResolutionRule.PRESET_FIRST:
            # Preset terms override document terms
            if preset_source:
                return "update_existing"
            return "keep_existing"

        elif self.conflict_rule == ConflictResolutionRule.DOCUMENT_INITIAL:
            # First occurrence wins
            if existing.source_chunk_indices:
                return "keep_existing"
            return "update_existing"

        elif self.conflict_rule == ConflictResolutionRule.MAJORITY:
            # Term with most occurrences wins
            existing_count = len(existing.source_chunk_indices)
            new_count = new_confidence * 10  # Convert confidence to count estimate

            if existing_count >= new_count:
                return "keep_existing"
            return "update_existing"

        elif self.conflict_rule == ConflictResolutionRule.MOST_RECENT:
            # Most recent occurrence wins
            if existing.source_chunk_indices:
                return "keep_existing"
            return "update_existing"

        else:
            # Default: majority rule
            existing_count = len(existing.source_chunk_indices)
            new_count = new_confidence * 10

            return "keep_existing" if existing_count >= new_count else "update_existing"

    def _apply_conflict_resolution(self, conflict: GlossaryConflict) -> str:
        """
        Apply conflict resolution to determine target term.

        Args:
            conflict: Conflict to resolve

        Returns:
            Resolved target term
        """
        if self.conflict_rule == ConflictResolutionRule.PRESET_FIRST:
            # Return first option (preset term)
            return conflict.options[0]

        elif self.conflict_rule == ConflictResolutionRule.DOCUMENT_INITIAL:
            # Return first option (document first occurrence)
            return conflict.options[0]

        elif self.conflict_rule == ConflictResolutionRule.MAJORITY:
            # Return option with highest count
            counts = {}
            for opt in conflict.options:
                counts[opt] = 0

            for opt in conflict.options:
                counts[opt] += 1

            # Return term with most occurrences
            return max(counts, key=counts.get)

        elif self.conflict_rule == ConflictResolutionRule.MOST_RECENT:
            # Return last option (most recent)
            return conflict.options[-1]

        else:
            # Default: majority
            counts = {}
            for opt in conflict.options:
                counts[opt] = 0

            for opt in conflict.options:
                counts[opt] += 1

            return max(counts, key=counts.get)

    def export_glossary(self) -> Dict[str, Any]:
        """
        Export glossary with metadata.

        Returns:
            Dict with glossary and metadata
        """
        return {
            "glossary": self.glossary_dict(),
            "conflicts_count": len(self._conflict_history),
            "resolution_rule": self.conflict_rule.value,
        }

    def glossary_dict(self) -> Dict[str, str]:
        """
        Get glossary as simple dictionary.

        Returns:
            Dict mapping source to target
        """
        return {
            term.source: term.target
            for term in self._get_all_terms()
        }

    def _get_all_terms(self) -> List[TermEntry]:
        """
        Get all term entries.

        Returns:
            List of term entries
        """
        # In real implementation, would include all glossaries
        return []
