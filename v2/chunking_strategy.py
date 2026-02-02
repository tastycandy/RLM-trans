"""
RLM Chunking Strategy
Provides intelligent chunking with semantic boundaries and overlap
"""
from typing import List, Tuple
import re
from rlm_state import ChunkPlan


class ChunkingStrategy:
    """
    Intelligent chunking strategy for translation.
    Preserves sentence/paragraph boundaries and supports overlap.
    """

    def __init__(self, chunk_size: int = 2000, overlap: int = 150):
        """
        Initialize chunking strategy.

        Args:
            chunk_size: Maximum chunk size in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Chunk text intelligently.

        Args:
            text: Text to chunk

        Returns:
            List of (start, end, chunk_text) tuples
        """
        if not text:
            return []

        chunks = []
        current_pos = 0
        text_length = len(text)

        while current_pos < text_length:
            # Find natural break point
            end = self._find_break_point(text, current_pos, self.chunk_size)

            chunk_text = text[current_pos:end].strip()

            if chunk_text:
                chunks.append((current_pos, end, chunk_text))

            # Move position forward
            if end >= text_length:
                break

            # Add overlap if available
            current_pos = max(current_pos + self.overlap, end)

        return chunks

    def chunk_by_paragraph(self, text: str, show_warning_callback=None) -> List[Tuple[int, int, str]]:
        """
        Chunk text by paragraph boundaries.
        Only includes complete paragraphs that fit within chunk_size.
        If a paragraph is larger than chunk_size, cuts at sentence boundary.

        Args:
            text: Text to chunk
            show_warning_callback: Optional callback function for warnings

        Returns:
            List of (start, end, chunk_text) tuples
        """
        if not text:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # Check if paragraph is too large
            if para_size > self.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append((current_start, current_start + len(chunk_text), chunk_text))
                    current_chunk = []
                    current_size = 0
                
                # Warn about large paragraph
                if show_warning_callback:
                    show_warning_callback(f"문단이 청크 크기({self.chunk_size}자)보다 큽니다. 문장 단위로 분할합니다.")
                
                # Split large paragraph by sentences
                sentence_chunks = self._split_paragraph_by_sentences(para)
                for sent_chunk in sentence_chunks:
                    chunks.append((0, len(sent_chunk), sent_chunk))
                
                current_start = 0
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_size + para_size + 2 > self.chunk_size:  # +2 for \n\n
                # Save current chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append((current_start, current_start + len(chunk_text), chunk_text))
                
                # Start new chunk with this paragraph
                current_chunk = [para]
                current_size = para_size
                current_start = 0
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size + 2
        
        # Save final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((current_start, current_start + len(chunk_text), chunk_text))
        
        return chunks
    
    def _split_paragraph_by_sentences(self, paragraph: str) -> List[str]:
        """
        Split a large paragraph into chunks at sentence boundaries.
        
        Args:
            paragraph: Large paragraph to split
            
        Returns:
            List of chunks, each ending at a sentence boundary
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?。！？])\s+', paragraph)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sent_size = len(sentence)
            
            if current_size + sent_size + 1 > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = sent_size
            else:
                current_chunk.append(sentence)
                current_size += sent_size + 1
        
        # Save final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def chunk_srt(self, srt_entries: List[dict]) -> List[Tuple[int, int, str]]:
        """
        Chunk SRT subtitle entries.

        Args:
            srt_entries: List of SRT entry dictionaries

        Returns:
            List of (start, end, chunk_text) tuples
        """
        chunks = []

        # Group entries into chunks
        current_chunk = []
        current_size = 0

        for entry in srt_entries:
            entry_text = entry.get('text', '')
            entry_size = len(entry_text)

            # Add entry if it fits
            if current_size + entry_size <= self.chunk_size:
                current_chunk.append(entry_text)
                current_size += entry_size
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    start = sum(len(srt_entries[i].get('text', '')) for i in range(
                        len(srt_entries) - len(current_chunk), len(srt_entries) - len(current_chunk) + len(current_chunk)
                    ))

                    chunks.append((0, 0, chunk_text))

                # Start new chunk with current entry
                current_chunk = [entry_text]
                current_size = entry_size

        # Save final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((0, 0, chunk_text))

        return chunks

    def chunk_patent(self, patent_text: str) -> List[Tuple[int, int, str]]:
        """
        Chunk patent document, preserving claims structure.

        Args:
            patent_text: Patent document text

        Returns:
            List of (start, end, chunk_text) tuples
        """
        chunks = []

        # Split by claim markers
        claim_pattern = r'(Claims?\d+[:.]|\(Claims?\d+\))'
        claims = re.split(claim_pattern, patent_text)

        current_chunk = ""
        current_start = 0

        for i in range(0, len(claims), 2):
            if i + 1 >= len(claims):
                break

            claim_marker = claims[i]
            claim_content = claims[i + 1]

            if claim_marker:
                # Save previous chunk if any
                if current_chunk.strip():
                    chunks.append((current_start, current_start + len(current_chunk), current_chunk.strip()))

                # Start new chunk with claim
                current_chunk = claim_marker + " " + claim_content
                current_start = len(patent_text) - len(claim_content)

        # Add last chunk if any
        if current_chunk.strip():
            chunks.append((current_start, len(patent_text), current_chunk.strip()))

        return chunks

    def _find_break_point(self, text: str, start: int, max_size: int) -> int:
        """
        Find natural break point within chunk size.

        Args:
            text: Full text
            start: Starting position
            max_size: Maximum chunk size

        Returns:
            Break position
        """
        # Try to find sentence boundary first
        end = self._find_sentence_boundary(text, start, max_size)

        if end > start:
            return end

        # Try paragraph boundary
        end = self._find_paragraph_boundary(text, start, max_size)

        if end > start:
            return end

        # Just break at max size
        return min(start + max_size, len(text))

    def _find_sentence_boundary(self, text: str, start: int, max_size: int) -> int:
        """
        Find sentence boundary.

        Args:
            text: Full text
            start: Starting position
            max_size: Maximum chunk size

        Returns:
            Sentence boundary position
        """
        # Find next period or question mark
        end = start

        while end < min(start + max_size, len(text) - 1):
            char = text[end]

            if char in '.!?':
                # Check if it's followed by space and maybe quote/paren
                next_char = text[end + 1] if end + 1 < len(text) else ''

                if next_char in ' "\')':
                    # Check for closing quote/paren
                    if char in '!?' and next_char in '"\' )':
                        # Skip quotes/parens after punctuation
                        end += 1
                        while end < len(text) and text[end] in '"\' )':
                            end += 1

                        if end < len(text) and text[end].isspace():
                            return end + 1
                    else:
                        return end + 1

            end += 1

        return start + max_size

    def _find_paragraph_boundary(self, text: str, start: int, max_size: int) -> int:
        """
        Find paragraph boundary.

        Args:
            text: Full text
            start: Starting position
            max_size: Maximum chunk size

        Returns:
            Paragraph boundary position
        """
        # Find double newline
        end = start

        while end < min(start + max_size, len(text) - 1):
            if text[end] == '\n' and end + 1 < len(text) and text[end + 1] == '\n':
                return end + 2

            end += 1

        return start + max_size

    def create_chunk_plan(
        self,
        chunks: List[Tuple[int, int, str]],
        strategy: str = "sequential"
    ) -> ChunkPlan:
        """
        Create chunk plan.

        Args:
            chunks: List of chunks
            strategy: Chunking strategy ('sequential', 'adaptive')

        Returns:
            ChunkPlan object
        """
        plan = ChunkPlan()

        plan.chunks = chunks.copy()
        plan.strategy = strategy

        # Assign priorities based on strategy
        if strategy == "adaptive":
            self._assign_adaptive_priorities(plan)

        return plan

    def _assign_adaptive_priorities(self, plan: ChunkPlan):
        """
        Assign priorities to chunks based on content complexity.

        Args:
            plan: Chunk plan to update
        """
        # Simple heuristic: longer chunks need more attention
        for i, (start, end, chunk) in enumerate(plan.chunks):
            length = end - start

            # Prioritize longer chunks
            priority = min(10, max(1, int(length / 500)))

            # Mark chunks as priority
            pass  # Priority info can be stored in chunk metadata

    def get_overlap_size(self, chunk1: str, chunk2: str) -> int:
        """
        Get overlap size between two chunks.

        Args:
            chunk1: First chunk
            chunk2: Second chunk

        Returns:
            Overlap size in characters
        """
        if not self.overlap:
            return 0

        # Simple word overlap
        words1 = set(chunk1.lower().split())
        words2 = set(chunk2.lower().split())

        overlap = words1 & words2

        return len(overlap) * 2  # Approximate

    def detect_content_type(self, text: str) -> str:
        """
        Detect content type (subtitle, patent, paper, general).

        Args:
            text: Sample text

        Returns:
            Content type string
        """
        text_lower = text.lower()

        # Check for subtitle markers
        if any(marker in text_lower for marker in ['[', ']', '---', '00:00:00']):
            return 'subtitle'

        # Check for patent markers
        if any(marker in text_lower for marker in ['claim', 'wherein', 'comprising']):
            return 'patent'

        # Check for academic markers
        if any(marker in text_lower for marker in ['abstract', 'introduction', 'conclusion', 'citation']):
            return 'paper'

        return 'general'
