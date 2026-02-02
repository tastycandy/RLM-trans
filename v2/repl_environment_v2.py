"""
RLM-Trans Enhanced REPL Environment
Safe context storage with whitelisted tools for Root Orchestrator
"""
import sys
import io
import traceback
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import re

from rlm_state import (
    TranslationState,
    PresetType,
    QualityFlagType,
    ChunkPlan,
    TermEntry,
    EntityEntry
)


@dataclass
class ChunkInfo:
    """Information about a chunk"""
    start: int
    end: int
    text: str
    index: int


class EnhancedREPL:
    """
    Enhanced REPL environment for RLM translation.
    Provides safe, whitelisted function calls for the Root Orchestrator.
    """

    def __init__(self, llm_query_func: Callable[[str], str],
                 preset_type: PresetType = PresetType.GENERAL):
        """
        Initialize REPL with LLM query function.

        Args:
            llm_query_func: Function to call sub-agent for translation
            preset_type: Current preset type
        """
        self.state = TranslationState(preset_id=preset_type)
        self._llm_query_func = llm_query_func
        self._final_result: Optional[str] = None
        self._is_finished = False
        self._output_buffer = io.StringIO()

        # Chunk information
        self._chunks: List[ChunkInfo] = []

        # Build execution namespace
        self._namespace = self._build_namespace()

    def _build_namespace(self) -> Dict[str, Any]:
        """Build the namespace for code execution with whitelisted functions"""
        return {
            # State variables
            'original_text': self.state.chunk_history,
            'translated_chunks': self.state.translation_history,
            'glossary': self._format_glossary_dict(),
            'entities': self._format_entities_dict(),
            'context_summary': self._get_context_summary(),

            # Whitelisted functions
            'get_chunk': self._get_chunk,
            'peek_chunks': self._peek_chunks,
            'get_all_chunks': self._get_all_chunks,
            'extract_terms': self._extract_terms,
            'update_glossary': self._update_glossary,
            'add_entity': self._add_entity,
            'summarize_context': self._summarize_context,
            'check_constraints': self._check_constraints,
            'compute_similarity': self._compute_similarity,
            'save_translation': self._save_translation,
            'select_next_chunk': self._select_next_chunk,
            'get_chunk_count': self._get_chunk_count,
            'get_preset_rules': self._get_preset_rules,
            'get_style_guide': self._get_style_guide,

            # Helpers
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'sorted': sorted,
            're': re,

            # Output
            'print': self._safe_print,
        }

    def _format_glossary_dict(self) -> Dict[str, str]:
        """Format glossary for REPL access (only non-hard entries for now)"""
        glossary_dict = {}
        for src, term in self.state.glossary.items():
            if not term.is_hard or term.usage_count > 5:
                glossary_dict[src] = term.target
        return glossary_dict

    def _format_entities_dict(self) -> Dict[str, str]:
        """Format entities for REPL access"""
        return {name: entity.translation for name, entity in self.state.entities.items()}

    def _get_context_summary(self) -> str:
        """Get current context summary"""
        if self.state.history_summaries:
            return '\n'.join(self.state.history_summaries)
        return "No context summary available yet."

    def set_preset_type(self, preset_type: PresetType):
        """Update preset type"""
        self.state.preset_id = preset_type
        self.state.document_type = preset_type.value

    def set_original_text(self, chunks: List[str]):
        """Set the original text chunks"""
        # Store original chunks separately - don't add to chunk_history yet
        self._original_chunks = chunks.copy()
        self._chunks = []
        for i, chunk in enumerate(chunks):
            start = sum(len(chunks[j]) for j in range(i))
            self._chunks.append(ChunkInfo(start, start + len(chunk), chunk, i))
        
        # Initialize empty histories - will be filled during translation
        self.state.chunk_history = []
        self.state.translation_history = []
        self.state.current_chunk_index = 0
        self.state.completed_chunks = 0

    def _get_chunk(self, chunk_index: int) -> Optional[str]:
        """Get a chunk by index"""
        if 0 <= chunk_index < len(self._chunks):
            return self._chunks[chunk_index].text
        return None

    def _peek_chunks(self, chunk_indices: List[int], context_lines: int = 3) -> str:
        """Get chunks with context lines around them"""
        result = []
        for idx in chunk_indices:
            if 0 <= idx < len(self._chunks):
                chunk = self._chunks[idx]
                result.append(f"=== Chunk {chunk.index} (start={chunk.start}, end={chunk.end}) ===")
                result.append(chunk.text)
                result.append("")
        return '\n'.join(result)

    def _get_all_chunks(self) -> str:
        """Get all chunks in order"""
        return '\n\n--- CHUNK SEPARATOR ---\n\n'.join(chunk.text for chunk in self._chunks)

    def _extract_terms(self, text: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract potential terms from text.

        Returns:
            List of dicts with 'source', 'target', 'confidence', 'context'
        """
        # Simple heuristic: words that appear multiple times or are capitalized
        terms = []
        word_freq = {}

        # Simple tokenization
        words = re.findall(r'\b\w+\b', text)

        # Count word frequency
        for word in words:
            if len(word) > 1:  # Skip single characters
                word_freq[word] = word_freq.get(word, 0) + 1

        # Find words with high frequency
        for word, count in sorted(word_freq.items(), key=lambda x: -x[1]):
            if count >= 2 and len(word) > 2:
                # Simple translation suggestion (placeholder)
                # In real implementation, this would use LLM or dictionary
                terms.append({
                    'source': word,
                    'target': word,  # Placeholder
                    'confidence': min(0.9, count / 10),
                    'context': text[:100]
                })

        return terms

    def _update_glossary(self, source: str, target: str, is_hard: bool = False):
        """Add or update glossary entry"""
        chunk_indices = []
        for i, chunk in enumerate(self.state.chunk_history):
            if source in chunk:
                chunk_indices.append(i)

        self.state.add_glossary_entry(
            source=source,
            target=target,
            confidence=0.7,
            source_chunk_indices=chunk_indices,
            is_hard=is_hard
        )

        self._safe_print(f"Glossary updated: {source} → {target} (hard={is_hard})")

    def _add_entity(self, name: str, translation: str, entity_type: str = "person"):
        """Add or update entity entry"""
        context = f"Appears in document with translation '{translation}'"
        self.state.add_entity(name, translation, entity_type, context)
        self._safe_print(f"Entity added: {name} → {translation} (type={entity_type})")

    def _summarize_context(self, last_k: int = 5) -> str:
        """Summarize recent translation context"""
        summaries = self.state.history_summaries[-last_k:]
        if not summaries:
            return "No context summaries available yet."

        return '\n'.join(f"  {i+1}. {summary}" for i, summary in enumerate(summaries))

    def _check_constraints(self, translation: str, preset_rules: List[str]) -> Dict[str, Any]:
        """
        Check translation against preset constraints.

        Returns:
            Dict with 'valid' (bool), 'issues' (list), 'warnings' (list)
        """
        issues = []
        warnings = []

        # Check for forbidden words/phrases
        for rule in preset_rules:
            if 'forbidden' in rule.lower() and rule in translation:
                issues.append(f"Forbidden phrase found: {rule}")

        # Check length constraints
        # (implementation depends on preset-specific rules)

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def _compute_similarity(self, prev_chunk: str, cur_chunk: str) -> float:
        """
        Compute similarity between two chunks.

        Returns:
            Float similarity score (0.0 to 1.0)
        """
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', prev_chunk.lower()))
        words2 = set(re.findall(r'\b\w+\b', cur_chunk.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _save_translation(self, chunk_index: int, translation: str, quality_flag: Optional[QualityFlagType] = None):
        """
        Save translation for a chunk.

        Args:
            chunk_index: Index of chunk to translate
            translation: Translated text
            quality_flag: Optional quality flag (for error tracking)
        """
        if 0 <= chunk_index < len(self.state.translation_history):
            self.state.update_chunk(chunk_index, translation)

            if quality_flag:
                error_msg = "Translation quality issue detected"
                self.state.record_error(chunk_index, quality_flag, error_msg)

            self._safe_print(f"Translation saved for chunk {chunk_index+1}/{len(self.state.translation_history)}")

    def _select_next_chunk(self, strategy: str = "sequential") -> Optional[int]:
        """
        Select the next chunk index based on strategy.

        Args:
            strategy: 'sequential', 'adaptive', 'priority'

        Returns:
            Next chunk index to process, or None if all done
        """
        if self.state.current_chunk_index >= len(self._chunks):
            return None

        chunk_indices = list(range(self.state.current_chunk_index, len(self._chunks)))

        if strategy == "sequential":
            return chunk_indices[0] if chunk_indices else None
        elif strategy == "adaptive":
            # Pick chunk with most context overlap (simple heuristic)
            if len(chunk_indices) >= 2:
                # Prefer the chunk with most words in common with previous
                prev_chunk = self.state.translation_history[-1] if self.state.translation_history else ""
                next_chunk = self.state.chunk_history[self.state.current_chunk_index]

                best_idx = 0
                best_sim = 0.0
                for i, idx in enumerate(chunk_indices):
                    sim = self._compute_similarity(prev_chunk, self._chunks[idx].text)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i
                return chunk_indices[best_idx]
            return chunk_indices[0] if chunk_indices else None
        else:
            return chunk_indices[0] if chunk_indices else None

    def _get_chunk_count(self) -> int:
        """Get total chunk count"""
        return len(self._chunks)

    def _get_preset_rules(self) -> List[str]:
        """Get preset-specific rules"""
        # In real implementation, this would load rules from preset
        return [
            "Translate preserving meaning and tone",
            "Use natural expressions in target language",
            "Maintain consistent terminology"
        ]

    def _get_style_guide(self) -> Dict[str, Any]:
        """Get style guide"""
        return {
            'tone': self.state.style_guide.tone,
            'politeness': self.state.style_guide.politeness,
            'sentence_length': self.state.style_guide.sentence_length
        }

    def _safe_print(self, *args, **kwargs):
        """Safe print that captures output"""
        output = ' '.join(str(arg) for arg in args)
        self._output_buffer.write(output + '\n')

    def execute(self, code: str, max_output_length: int = 500000) -> str:
        """
        Execute Python code in the REPL environment.

        Returns: Output from execution (stdout/stderr)
        """
        # Reset output buffer
        self._output_buffer = io.StringIO()

        # Update namespace with current state
        self._namespace['original_text'] = self.state.chunk_history
        self._namespace['translated_chunks'] = self.state.translation_history
        self._namespace['glossary'] = self._format_glossary_dict()
        self._namespace['entities'] = self._format_entities_dict()
        self._namespace['context_summary'] = self._get_context_summary()

        try:
            # Extract code from markdown code blocks if present
            code = self._extract_code(code)

            # Execute code
            exec(code, self._namespace)

        except Exception as e:
            self._safe_print(f"Error: {type(e).__name__}: {e}")
            self._safe_print(traceback.format_exc())

        output = self._output_buffer.getvalue()

        # Truncate if too long
        if len(output) > max_output_length:
            output = output[:max_output_length] + "\n... (output truncated)"

        return output

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks"""
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return '\n'.join(matches)

        return text

    @property
    def is_finished(self) -> bool:
        """Check if FINAL has been called"""
        return self._is_finished

    @property
    def final_result(self) -> Optional[str]:
        """Get the final result"""
        return self._final_result

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for LLM context"""
        return {
            'text_length': sum(len(chunk) for chunk in self.state.chunk_history),
            'chunks_translated': self.state.completed_chunks,
            'chunks_total': self.state.total_chunks,
            'glossary_size': len(self.state.glossary),
            'entities_size': len(self.state.entities),
            'history_summaries_count': len(self.state.history_summaries),
            'current_chunk': self.state.current_chunk_index,
        }

    def set_final_result(self, result: str):
        """Set the final result and mark as finished"""
        self._final_result = result
        self._is_finished = True
        self._safe_print("FINAL result set.")

    def get_translated_text(self) -> str:
        """Get all translated chunks concatenated"""
        return ''.join(self.state.translation_history)

    def reset(self):
        """Reset the REPL environment"""
        self.state.reset()
        self._final_result = None
        self._is_finished = False
        self._chunks = []
        self._namespace = self._build_namespace()
