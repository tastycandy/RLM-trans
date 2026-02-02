"""
RLM Translation State Management
Manages translation project memory with comprehensive state tracking
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class PresetType(str, Enum):
    """Available translation preset types"""
    SUBTITLE = "subtitle"
    PATENT = "patent"
    PAPER = "paper"
    NOVEL = "novel"
    TECHNICAL = "technical"
    GENERAL = "general"


class QualityFlagType(str, Enum):
    """Quality issue types"""
    FORMAT_ERROR = "format_error"
    MISSING_CONTENT = "missing_content"
    FORBIDDEN_WORD = "forbidden_word"
    TERMINOLOGY_MISMATCH = "terminology_mismatch"
    TOO_LONG = "too_long"
    MEANING_LOST = "meaning_lost"
    TONE_MISMATCH = "tone_mismatch"
    DUPLICATE_CONTENT = "duplicate_content"


class RepairType(str, Enum):
    """Repair action types"""
    TEMPLATE_REINFORCE = "template_reinforce"
    GLOSSARY_UPDATE = "glossary_update"
    SPLIT_CHUNK = "split_chunk"
    RE_TRANSLATE = "re_translate"
    CONTEXT_ADJUST = "context_adjust"


@dataclass
class ChunkPlan:
    """Chunk planning information"""
    chunks: List[tuple] = field(default_factory=list)  # [(start, end, chunk_text), ...]
    current_index: int = 0
    overlap: int = 0  # Number of characters/lines to overlap
    strategy: str = "semantic"  # semantic, sequential, adaptive


@dataclass
class TermEntry:
    """Glossary term entry"""
    source: str
    target: str
    confidence: float = 0.0
    source_chunk_indices: List[int] = field(default_factory=list)
    is_hard: bool = False  # Must be enforced in translation
    usage_count: int = 0


@dataclass
class EntityEntry:
    """Named entity entry"""
    name: str
    translation: str
    type: str = "person"  # person, place, organization, product
    context: str = ""
    usage_count: int = 0


@dataclass
class QualityFlags:
    """Translation quality tracking"""
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    retry_count: Dict[str, int] = field(default_factory=dict)  # flag_type -> count
    error_chunks: List[tuple] = field(default_factory=list)  # (chunk_index, error_type, message)
    quality_score: float = 0.0


@dataclass
class CostStats:
    """Cost and performance tracking"""
    root_calls: int = 0
    sub_calls: int = 0
    verifier_calls: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0  # seconds


@dataclass
class StyleGuide:
    """Translation style guide"""
    tone: str = "neutral"  # formal, informal, academic, conversational
    politeness: str = "default"  # honorific, plain, no honorifics
    sentence_length: str = "balanced"  # short, balanced, long
    forbidden_words: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)
    custom_rules: List[str] = field(default_factory=list)


@dataclass
class TranslationState:
    """Complete translation project memory"""
    # Core metadata
    preset_id: PresetType = PresetType.GENERAL
    document_type: str = "general"

    # Chunking
    chunk_plan: ChunkPlan = field(default_factory=ChunkPlan)
    chunk_history: List[str] = field(default_factory=list)  # Original chunks
    translation_history: List[str] = field(default_factory=list)  # Translated chunks

    # Glossary and entities
    glossary: Dict[str, TermEntry] = field(default_factory=dict)
    entities: Dict[str, EntityEntry] = field(default_factory=dict)
    
    # Term management (용어 관리 고도화)
    term_candidates: Dict[str, str] = field(default_factory=dict)  # 서브 에이전트가 제안한 용어 후보
    confirmed_terms: Dict[str, str] = field(default_factory=dict)  # 루트가 확정한 용어
    
    # Hard/Soft glossary separation (일관성 강제)
    hard_glossary: Dict[str, str] = field(default_factory=dict)  # 필수 준수 용어 (도면부호, 고유명사)
    soft_glossary: Dict[str, str] = field(default_factory=dict)  # 참고용 용어 (권장)
    
    # Named entity categories (분류별 관리)
    proper_nouns: Dict[str, str] = field(default_factory=dict)  # 고유명사 (인명, 지명, 상표)
    reference_signs: Dict[str, str] = field(default_factory=dict)  # 도면 부호 (100: 제어부)
    technical_terms: Dict[str, str] = field(default_factory=dict)  # 기술 명사 (Member, Assembly)

    # Style guide
    style_guide: StyleGuide = field(default_factory=StyleGuide)

    # Context management (sliding window)
    history_summaries: List[str] = field(default_factory=list)
    max_history_summaries: int = 5  # Keep last 5 summaries

    # Quality tracking
    quality_flags: QualityFlags = field(default_factory=QualityFlags)

    # Cost and performance
    cost_stats: CostStats = field(default_factory=CostStats)

    # Meta
    total_chunks: int = 0
    completed_chunks: int = 0
    current_chunk_index: int = 0

    def add_chunk(self, chunk_text: str, translation: str):
        """Add a new chunk to translation history"""
        self.chunk_history.append(chunk_text)
        self.translation_history.append(translation)
        self.current_chunk_index += 1
        self.completed_chunks += 1
        self.quality_flags.completed_chunks += 1

    def update_chunk(self, chunk_index: int, translation: str):
        """Update translation of existing chunk (for repair)"""
        if 0 <= chunk_index < len(self.translation_history):
            self.translation_history[chunk_index] = translation

    def add_glossary_entry(self, source: str, target: str, confidence: float = 0.0,
                          source_chunk_indices: Optional[List[int]] = None,
                          is_hard: bool = False):
        """Add or update glossary entry"""
        term = self.glossary.get(source)

        if term:
            # Update existing entry
            term.target = target
            term.confidence = max(term.confidence, confidence)
            if source_chunk_indices:
                term.source_chunk_indices.extend(source_chunk_indices)
            term.usage_count += 1
        else:
            # Create new entry
            term = TermEntry(
                source=source,
                target=target,
                confidence=confidence,
                source_chunk_indices=source_chunk_indices or [],
                is_hard=is_hard
            )
            self.glossary[source] = term

    def add_entity(self, name: str, translation: str, entity_type: str = "person",
                  context: str = ""):
        """Add or update entity entry"""
        entity = self.entities.get(name)

        if entity:
            entity.translation = translation
            entity.type = entity_type
            entity.context = context
            entity.usage_count += 1
        else:
            entity = EntityEntry(name=name, translation=translation,
                               type=entity_type, context=context)
            self.entities[name] = entity

    def add_history_summary(self, summary: str):
        """Add context summary with sliding window"""
        self.history_summaries.append(summary)

        # Keep only last N summaries
        if len(self.history_summaries) > self.max_history_summaries:
            self.history_summaries = self.history_summaries[-self.max_history_summaries:]

    def increment_retry_count(self, flag_type: QualityFlagType):
        """Increment retry counter for a quality issue type"""
        if not hasattr(self.quality_flags.retry_count, flag_type):
            self.quality_flags.retry_count[flag_type] = 0
        self.quality_flags.retry_count[flag_type] += 1

    def record_error(self, chunk_index: int, error_type: QualityFlagType, message: str):
        """Record a translation error"""
        self.quality_flags.error_chunks.append((chunk_index, error_type.value, message))
        self.quality_flags.failed_chunks += 1

    def get_hard_glossary(self, top_n: int = 50) -> Dict[str, str]:
        """Get top N hard glossary entries"""
        hard_terms = sorted(
            self.glossary.items(),
            key=lambda x: (x[1].usage_count if x[1].usage_count > 0 else -1,
                          x[1].confidence if x[1].confidence > 0 else 0),
            reverse=True
        )[:top_n]

        return {src: term.target for src, term in hard_terms}

    def get_entity_translations(self, top_n: int = 100) -> Dict[str, str]:
        """Get top N entity translations"""
        entities = sorted(
            self.entities.items(),
            key=lambda x: x[1].usage_count if x[1].usage_count > 0 else 0,
            reverse=True
        )[:top_n]

        return {name: entity.translation for name, entity in entities}

    # === 용어 관리 도구 함수 (update.md 1단계) ===
    
    def propose_terms(self, term_candidates: Dict[str, str]):
        """
        서브 에이전트가 발견한 새 용어를 후보군에 추가.
        기존에 있는 용어와 충돌하면 후보로만 저장.
        """
        for source, target in term_candidates.items():
            # 이미 확정된 용어가 있으면 후보에만 추가
            if source not in self.confirmed_terms:
                self.term_candidates[source] = target
    
    def update_glossary(self, source: str, target: str, force: bool = True):
        """
        루트 에이전트가 후보 중 적절한 것을 골라 정식 용어집에 등록.
        force=True면 기존 용어도 덮어씀.
        """
        if force or source not in self.confirmed_terms:
            self.confirmed_terms[source] = target
            # 후보에서 제거
            if source in self.term_candidates:
                del self.term_candidates[source]
    
    def add_hard_term(self, source: str, target: str):
        """필수 준수 용어 추가 (도면부호, 고유명사)"""
        self.hard_glossary[source] = target
        self.confirmed_terms[source] = target
    
    def add_soft_term(self, source: str, target: str):
        """참고용 용어 추가 (권장)"""
        self.soft_glossary[source] = target
    
    def add_proper_noun(self, source: str, target: str):
        """고유명사 추가 (인명, 지명, 상표)"""
        self.proper_nouns[source] = target
        self.add_hard_term(source, target)
    
    def add_reference_sign(self, sign: str, meaning: str):
        """도면 부호 추가 (예: 100 -> 제어부)"""
        self.reference_signs[sign] = meaning
        self.add_hard_term(sign, meaning)
    
    def add_technical_term(self, source: str, target: str, is_hard: bool = False):
        """기술 명사 추가"""
        self.technical_terms[source] = target
        if is_hard:
            self.add_hard_term(source, target)
        else:
            self.add_soft_term(source, target)
    
    def get_hard_glossary_text(self) -> str:
        """
        다음 번역 시 서브 에이전트에게 강제할 '필수 용어 리스트'를 텍스트 형태로 추출.
        """
        lines = ["=== 필수 준수 용어 (Hard Glossary) ==="]
        
        # 확정 용어
        for src, tgt in self.hard_glossary.items():
            lines.append(f"  {src} → {tgt}")
        
        # 도면 부호
        if self.reference_signs:
            lines.append("\n[도면 부호]")
            for sign, meaning in self.reference_signs.items():
                lines.append(f"  {sign}: {meaning}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def get_soft_glossary_text(self) -> str:
        """참고용 용어 리스트 텍스트"""
        lines = ["=== 권장 용어 (Soft Glossary) ==="]
        for src, tgt in self.soft_glossary.items():
            lines.append(f"  {src} → {tgt}")
        return "\n".join(lines) if len(lines) > 1 else ""
    
    def get_context_package(self) -> Dict[str, Any]:
        """
        서브 에이전트가 번역 전 받는 전체 컨텍스트 패키지.
        Hard/Soft Glossary, 스타일 가이드 포함.
        """
        return {
            "hard_glossary": self.hard_glossary.copy(),
            "soft_glossary": self.soft_glossary.copy(),
            "proper_nouns": self.proper_nouns.copy(),
            "reference_signs": self.reference_signs.copy(),
            "technical_terms": self.technical_terms.copy(),
            "confirmed_terms": self.confirmed_terms.copy(),
            "style_guide": {
                "tone": self.style_guide.tone,
                "forbidden_words": self.style_guide.forbidden_words,
            },
            "history_summaries": self.history_summaries[-3:] if self.history_summaries else [],
        }
    
    def check_term_conflict(self, source: str, new_target: str) -> Optional[str]:
        """
        용어 충돌 확인. 기존 용어와 다른 번역어가 제안되면 기존 값 반환.
        """
        if source in self.confirmed_terms:
            existing = self.confirmed_terms[source]
            if existing != new_target:
                return existing
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get state summary for debugging/monitoring"""
        return {
            "preset_id": self.preset_id,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "glossary_size": len(self.glossary),
            "entities_size": len(self.entities),
            "history_summaries_count": len(self.history_summaries),
            "failed_chunks": self.quality_flags.failed_chunks,
            "total_retries": sum(self.quality_flags.retry_count.values()),
            "quality_score": self.quality_flags.quality_score,
            "total_cost": self.cost_stats.total_cost,
        }

    def reset(self):
        """Reset state to initial state"""
        self.chunk_plan = ChunkPlan()
        self.chunk_history.clear()
        self.translation_history.clear()
        self.glossary.clear()
        self.entities.clear()
        self.history_summaries.clear()
        self.quality_flags = QualityFlags()
        self.cost_stats = CostStats()
        self.current_chunk_index = 0
        self.completed_chunks = 0
        
        # Reset term management
        self.term_candidates.clear()
        self.confirmed_terms.clear()
        self.hard_glossary.clear()
        self.soft_glossary.clear()
        self.proper_nouns.clear()
        self.reference_signs.clear()
        self.technical_terms.clear()
