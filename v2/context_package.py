"""
RLM Context Package Builder
Creates structured context package for sub-translator
"""
from typing import Dict, Any, List
from rlm_state import TranslationState, PresetType


def build_context_package(
    state: TranslationState,
    current_chunk_text: str,
    current_chunk_index: int,
    hard_glossary: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Build context package for sub-translator.

    Args:
        state: Translation state
        current_chunk_text: Text of current chunk to translate
        current_chunk_index: Index of current chunk
        hard_glossary: Additional hard glossary entries (optional)

    Returns:
        Dict with structured context
    """
    # Base package from state (includes Hard/Soft terms, style guide)
    package = state.get_context_package()

    # Override/Extend hard glossary if provided explicitly
    if hard_glossary:
        package["hard_glossary"].update(hard_glossary)

    # Build local context (last 3-5 chunks)
    local_context = _build_local_context(state)

    # Build rules section from preset
    rules = _build_rules(state.preset_id)

    # Merge into package
    package.update({
        "rules": rules,
        "local_context": local_context,
        "chunk": current_chunk_text,
        "chunk_index": current_chunk_index,
        "document_type": state.document_type,
    })

    return package


def _build_rules(preset_type: PresetType) -> List[str]:
    """Build rules based on preset type"""
    rules = [
        "Translate preserving meaning and intent",
        "Use natural expressions in target language",
        "Maintain consistent terminology throughout"
    ]

    # Add preset-specific rules
    if preset_type == PresetType.SUBTITLE:
        rules.extend([
            "Keep translations SHORT and natural for spoken dialogue",
            "Match timing constraints of subtitles",
            "Use colloquial expressions appropriate for speech",
            "Avoid overly formal language",
            "Keep line breaks where they make sense for readability"
        ])
    elif preset_type == PresetType.PATENT:
        rules.extend([
            "Use EXACT legal terminology - precision is critical",
            "Maintain claim structure and numbering",
            "Preserve all technical specifications exactly",
            "Keep patent-specific phrases (comprising, wherein)",
            "Do not paraphrase - translate literally as appropriate",
            "Maintain reference numbers and figure references"
        ])
    elif preset_type == PresetType.PAPER:
        rules.extend([
            "Use precise academic terminology",
            "Maintain formal, objective tone",
            "Preserve technical terms (transliterate if no standard translation)",
            "Keep citation formats intact",
            "Translate figure/table captions accurately",
            "Maintain logical flow and argumentation structure"
        ])
    elif preset_type == PresetType.NOVEL:
        rules.extend([
            "Preserve author's unique voice and style",
            "Maintain narrative flow and pacing",
            "Translate idioms naturally, not literally",
            "Keep character voice distinctions",
            "Preserve metaphors and literary devices when possible",
            "Adapt cultural references appropriately",
            "Maintain emotional impact and atmosphere"
        ])
    elif preset_type == PresetType.TECHNICAL:
        rules.extend([
            "Use clear, unambiguous language",
            "Maintain consistent terminology",
            "Preserve code snippets and commands exactly",
            "Keep formatting (lists, headings, tables)",
            "Translate UI text according to localization standards",
            "Keep placeholder text unchanged"
        ])

    return rules


def _build_style_guide(style_guide) -> Dict[str, Any]:
    """Build style guide section"""
    return {
        "tone": style_guide.tone,
        "politeness": style_guide.politeness,
        "sentence_length": style_guide.sentence_length,
        "forbidden_words": style_guide.forbidden_words,
        "forbidden_phrases": style_guide.forbidden_phrases,
    }


def _build_local_context(state: TranslationState) -> Dict[str, Any]:
    """Build local context from recent history"""
    # Get last 3-5 chunks
    recent_translations = state.translation_history[-5:] if state.translation_history else []
    recent_originals = state.chunk_history[-5:] if state.chunk_history else []

    # Build summary
    summaries = state.history_summaries[-5:] if state.history_summaries else []

    return {
        "recent_translations": recent_translations[-3:],  # Last 3 translations
        "recent_originals": recent_originals[-3:],  # Last 3 originals
        "context_summaries": summaries,  # Context summaries
        "entity_translations": state.get_entity_translations(),  # Entity mappings
    }


def get_context_package_string(package: Dict[str, Any]) -> str:
    """
    Convert context package to string for LLM input.

    Returns: Formatted string ready for LLM prompt
    """
    lines = []

    lines.append("=== CONTEXT PACKAGE ===")
    lines.append("")

    lines.append("RULES:")
    for rule in package.get("rules", []):
        lines.append(f"  - {rule}")
    lines.append("")

    lines.append("GLOSSARY (Hard - Must Use):")
    for src, target in package.get("glossary", []):
        lines.append(f"  - {src} → {target}")
    lines.append("")

    lines.append("STYLE GUIDE:")
    style = package.get("style", {})
    lines.append(f"  - Tone: {style.get('tone', 'neutral')}")
    lines.append(f"  - Politeness: {style.get('politeness', 'default')}")
    lines.append(f"  - Sentence Length: {style.get('sentence_length', 'balanced')}")
    lines.append("")

    lines.append("LOCAL CONTEXT:")
    local_ctx = package.get("local_context", {})
    lines.append(f"  - Document Type: {package.get('document_type', 'general')}")
    lines.append(f"  - Recent Translations: {len(local_ctx.get('recent_translations', []))} chunks")
    lines.append(f"  - Entity Mappings: {len(local_ctx.get('entity_translations', {}))} entities")
    lines.append("")

    lines.append("CURRENT CHUNK TO TRANSLATE:")
    lines.append(f"  - Index: {package.get('chunk_index', 0)}")
    lines.append(f"  - Text: {package.get('chunk', '')[:500]}")
    lines.append("")

    lines.append("=== END OF CONTEXT PACKAGE ===")
    lines.append("")

    return "\n".join(lines)


def get_translation_instructions(package: Dict[str, Any]) -> str:
    """
    Get translation instructions for LLM.

    Returns: Instructions on how to translate the chunk
    """
    lines = [
        "=== TRANSLATION INSTRUCTIONS ===",
        "",
        "Please translate the CURRENT CHUNK using the context and rules above.",
        "",
        "Requirements:",
        "1. Follow all rules specified above",
        "2. Use the glossary entries where applicable",
        "3. Match the style guide (tone, politeness, sentence length)",
        "4. Consider the local context (previous translations, entities)",
        "5. Maintain consistency with existing translations",
        "",
        "Output format: Provide ONLY the translated text, no explanations.",
        "",
        "=== END ==="
    ]

    return "\n".join(lines)
