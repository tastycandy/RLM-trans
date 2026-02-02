"""
RLM-Trans Text Utilities
Language detection, text chunking, and format handling
"""
import re
from typing import List, Tuple, Optional


def detect_language(text: str) -> str:
    """
    Detect the language of the text.
    Returns: 'ko', 'ja', 'en', or 'unknown'
    """
    # Sample the text (first 1000 chars for efficiency)
    sample = text[:1000]
    
    # Count character types
    hangul_count = len(re.findall(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', sample))
    japanese_count = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', sample))  # Hiragana + Katakana
    kanji_count = len(re.findall(r'[\u4E00-\u9FFF]', sample))  # CJK (shared by Japanese/Chinese)
    latin_count = len(re.findall(r'[a-zA-Z]', sample))
    
    total = hangul_count + japanese_count + kanji_count + latin_count
    
    if total == 0:
        return "unknown"
    
    # Korean: primarily Hangul
    if hangul_count > total * 0.3:
        return "ko"
    
    # Japanese: Hiragana/Katakana present, possibly with Kanji
    if japanese_count > 0 or (kanji_count > 0 and hangul_count == 0 and latin_count < kanji_count):
        if japanese_count > 0:
            return "ja"
        # If only Kanji, could be Chinese or Japanese - check for patterns
        # For simplicity, if no Hangul and has Kanji with some Latin, assume Japanese
        if kanji_count > total * 0.2:
            return "ja"
    
    # English: primarily Latin characters
    if latin_count > total * 0.5:
        return "en"
    
    return "unknown"


def chunk_text(text: str, chunk_size: int = 2000, 
               overlap: int = 100) -> List[Tuple[int, int, str]]:
    """
    Split text into chunks at sentence boundaries.
    
    Returns: List of (start_index, end_index, chunk_text)
    """
    if len(text) <= chunk_size:
        return [(0, len(text), text)]
    
    chunks = []
    start = 0
    
    # Sentence ending patterns for different languages
    sentence_endings = re.compile(r'[.!?。！？]\s*|\n\n+')
    
    while start < len(text):
        # Calculate end of this chunk
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            # Try to find a sentence boundary
            chunk = text[start:end]
            
            # Search backwards for sentence ending
            matches = list(sentence_endings.finditer(chunk))
            
            if matches:
                # Use the last sentence boundary in the chunk
                last_match = matches[-1]
                end = start + last_match.end()
            else:
                # Fallback: try to split at newline or space
                last_newline = chunk.rfind('\n')
                if last_newline > chunk_size * 0.5:
                    end = start + last_newline + 1
                else:
                    last_space = chunk.rfind(' ')
                    if last_space > chunk_size * 0.5:
                        end = start + last_space + 1
        
        chunk_text = text[start:end]
        chunks.append((start, end, chunk_text))
        
        # Move start with overlap for context continuity
        start = max(end - overlap, end - 50) if end < len(text) else end
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text for translation"""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace while preserving paragraph structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def is_srt_format(text: str) -> bool:
    """Check if text is SRT subtitle format"""
    # SRT format: number, timestamp, text, blank line
    pattern = r'^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
    return bool(re.search(pattern, text, re.MULTILINE))


def parse_srt(text: str) -> List[dict]:
    """Parse SRT subtitle file"""
    entries = []
    blocks = re.split(r'\n\n+', text.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            try:
                # First line: index
                index = int(lines[0].strip())
                
                # Second line: timestamp
                timestamp_match = re.match(
                    r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                    lines[1]
                )
                
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    
                    # Remaining lines: subtitle text
                    subtitle_text = '\n'.join(lines[2:]) if len(lines) > 2 else ""
                    
                    entries.append({
                        'index': index,
                        'start': start_time,
                        'end': end_time,
                        'text': subtitle_text
                    })
            except (ValueError, IndexError):
                continue
    
    return entries


def format_srt(entries: List[dict]) -> str:
    """Format subtitle entries back to SRT format"""
    blocks = []
    for entry in entries:
        block = f"{entry['index']}\n{entry['start']} --> {entry['end']}\n{entry['text']}"
        blocks.append(block)
    return '\n\n'.join(blocks) + '\n'


def extract_terms(text: str, min_freq: int = 2) -> List[str]:
    """Extract potential terms for glossary (names, repeated terms)"""
    # Find capitalized words (potential names)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Find quoted terms
    quoted = re.findall(r'[「」『』""''](.*?)[」』"'']', text)
    
    # Count occurrences
    from collections import Counter
    name_counts = Counter(names)
    
    # Return terms that appear multiple times
    frequent_terms = [term for term, count in name_counts.items() if count >= min_freq]
    
    return list(set(frequent_terms + quoted))
