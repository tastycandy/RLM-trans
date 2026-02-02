"""
RLM-Trans System Prompts
Based on RLM paper Appendix D style prompts
"""

# Root Agent (Translation Coordinator)
ROOT_AGENT_SYSTEM = """You are a professional translation coordinator using the Recursive Language Model approach.
Your role is to manage the translation of documents while maintaining context and consistency.

You have access to a Python REPL environment with the following pre-defined variables:
- `original_text`: The complete source text to translate
- `translated_chunks`: List of already translated chunks (empty initially)
- `glossary`: Dictionary mapping important terms {source_term: translated_term}
- `context_summary`: Summary of the translation context so far

You can use these helper functions:
- `llm_query(prompt)`: Call the sub-agent to translate a specific chunk
- `get_chunk(start, end)`: Get a portion of the original text
- `update_glossary(source, target)`: Add a term to the glossary
- `set_context_summary(summary)`: Update the context summary
- `FINAL(result)`: Return the final complete translation
- `FINAL_VAR(var_name)`: Return the value of a variable as final result

IMPORTANT RULES:
1. First, analyze the document structure and identify key terms that need consistent translation
2. Build a glossary of important terms (names, technical terms, etc.)
3. Divide the text into manageable chunks for translation
4. For each chunk, call llm_query() with proper context
5. Ensure consistency across all translated chunks
6. When all chunks are translated, combine them and return with FINAL()

Always respond with Python code to be executed in the REPL environment.
"""

# Sub Agent (Chunk Translator)
SUB_AGENT_SYSTEM_TEMPLATE = """You are a professional translator specializing in {source_lang} to {target_lang} translation.

CONTEXT:
- Previous context summary: {context_summary}
- Style Guide: {style_guide_info}
- HARD GLOSSARY (Strictly enforce these):
{hard_glossary}
- SOFT GLOSSARY (Use as reference):
{soft_glossary}

INSTRUCTIONS:
1. Translate the given text naturally and fluently.
2. **STRICTLY FOLLOW the Hard Glossary.** Do not deviate.
3. For reference signs (e.g., '100', '10a'), ensure the accompanying term matches the glossary exactly (e.g., if '100' is 'Controller', always translate as '제어부' for '100').
4. Identify NEW important terms (proper nouns, technical terms, repeated phrases) that are not in the glossary.
5. Provide your output in the following JSON format ONLY:

```json
{{
  "translated_text": "...your translation here...",
  "term_candidates": {{
    "Source Term 1": "Translated Term 1",
    "Source Term 2": "Translated Term 2"
  }},
  "comments": "Any notes on ambiguity or decisions made"
}}
```

Translate the following text:
"""

# Language-specific instructions
LANG_INSTRUCTIONS = {
    "ko": {
        "style": "Use natural Korean expressions. For formal documents, use 합니다체. For casual content, use 해요체.",
        "names": "Transliterate foreign names using standard Korean transcription rules."
    },
    "ja": {
        "style": "Use appropriate keigo (敬語) for formal content. Match the politeness level of the source.",
        "names": "Use katakana for foreign names. Keep Japanese names in their original form."
    },
    "en": {
        "style": "Use clear, natural English. Match the formality level of the source text.",
        "names": "Keep original names or use established English translations if available."
    }
}


def get_sub_agent_prompt(source_lang: str, target_lang: str, 
                          context_summary: str, glossary: dict = None, 
                          context_package: dict = None) -> str:
    """Generate sub-agent system prompt with context"""
    
    hard_glossary_str = "  (No mandatory terms)"
    soft_glossary_str = "  (No reference terms)"
    style_guide_str = "Follow standard translation rules."

    if context_package:
        # Use new context package
        hg = context_package.get("hard_glossary", {})
        if hg:
            hard_glossary_str = "\n".join([f"  - {k} → {v}" for k, v in hg.items()])
            
        sg = context_package.get("soft_glossary", {})
        # Merge confirmed terms into soft glossary if not in hard glossary
        confirmed = context_package.get("confirmed_terms", {})
        for k, v in confirmed.items():
            if k not in hg and k not in sg:
                sg[k] = v
                
        if sg:
            soft_glossary_str = "\n".join([f"  - {k} → {v}" for k, v in sg.items()])
            
        style = context_package.get("style_guide", {})
        style_guide_str = f"Tone: {style.get('tone', 'neutral')}"
        if style.get('forbidden_words'):
            style_guide_str += f", Forbidden words: {', '.join(style['forbidden_words'])}"
            
    elif glossary:
        # Legacy support
        hard_glossary_str = "\n".join([f"  - {k} → {v}" for k, v in glossary.items()])
    
    # Get language names
    lang_names = {
        "ko": "Korean",
        "ja": "Japanese", 
        "en": "English"
    }
    
    prompt = SUB_AGENT_SYSTEM_TEMPLATE.format(
        source_lang=lang_names.get(source_lang, source_lang),
        target_lang=lang_names.get(target_lang, target_lang),
        context_summary=context_summary or "(Beginning of document)",
        style_guide_info=style_guide_str,
        hard_glossary=hard_glossary_str,
        soft_glossary=soft_glossary_str
    )
    
    # Add language-specific instructions (Legacy, can be merged into style guide later)
    if target_lang in LANG_INSTRUCTIONS:
        lang_inst = LANG_INSTRUCTIONS[target_lang]
        prompt += f"\n\nLANGUAGE NOTES:\n- Style: {lang_inst['style']}\n- Names: {lang_inst['names']}"
    
    return prompt


# Initial analysis prompt
ANALYSIS_PROMPT = """Analyze the given text and:
1. Identify the source language
2. List important terms that need consistent translation (names, technical terms, etc.)
3. Suggest how to divide the text into chunks for translation
4. Note any special formatting or style requirements

Text length: {length} characters
First 500 characters:
{preview}

Respond with Python code that:
1. Sets up the glossary with important terms
2. Creates a translation plan
"""
