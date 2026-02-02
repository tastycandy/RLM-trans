"""
RLM Sub Translator
Handles translation of individual chunks with context package
"""
import json
import re
from typing import Dict, Any, List, Optional
import time

from config import LLMConfig
from llm_client import LLMClient
from rlm_state import PresetType
from context_package import build_context_package, get_context_package_string
# prompts.py에서 get_sub_agent_prompt 가져오기
from prompts import get_sub_agent_prompt


class SubTranslator:
    """
    Sub-translator agent that handles individual chunk translation.
    Works with context package for consistency.
    """

    def __init__(self, llm_config: LLMConfig, preset_type: PresetType = PresetType.GENERAL,
                 source_lang: str = "auto", target_lang: str = "ko"):
        """
        Initialize sub-translator.

        Args:
            llm_config: LLM provider configuration
            preset_type: Translation preset type
            source_lang: Source language code
            target_lang: Target language code
        """
        self.llm_config = llm_config
        self.preset_type = preset_type
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.llm_client = LLMClient(llm_config)

    def translate_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        state: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Translate a single chunk with context package.

        Args:
            chunk_text: Text to translate
            chunk_index: Index of chunk (for context)
            state: Translation state (for glossary, entities, etc.)

        Returns:
            Dict with 'translation', 'term_candidates', 'warnings', 'success'
        """
        start_time = time.time()

        try:
            # Build context package
            context_package = build_context_package(
                state=state,
                current_chunk_text=chunk_text,
                current_chunk_index=chunk_index
            )

            # Build messages for LLM using prompts.py
            messages = self._build_messages(context_package)

            # Call LLM
            response = self.llm_client.complete(
                messages,
                is_sub_call=True,
                max_tokens=8192
            )
            
            # Parse response (JSON or Text fallback)
            result = self._parse_llm_response(response.content, chunk_text)
            
            # Calculate duration
            duration = time.time() - start_time

            return {
                "translation": result["translation"],
                "term_candidates": result["term_candidates"],
                "warnings": result.get("warnings", []),
                "success": True,
                "duration": duration,
                "token_usage": self.llm_client.cost_summary() if hasattr(self.llm_client, 'cost_summary') else {},
            }

        except Exception as e:
            duration = time.time() - start_time
            import traceback
            traceback.print_exc()

            return {
                "translation": "",
                "term_candidates": {},
                "warnings": [f"Translation failed: {str(e)}"],
                "success": False,
                "duration": duration,
                "error": str(e),
            }

    def _build_messages(self, context_package: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build messages for LLM using prompts.py"""
        messages = []
        
        # Determine source language (if auto, use context or default)
        src_lang = self.source_lang if self.source_lang != "auto" else "en" 
        
        # Get System Prompt from prompts.py
        # context_package['history_summaries'] is list strings, prompts expects string
        context_summary = "\n".join(context_package.get('history_summaries', []))
        
        system_prompt = get_sub_agent_prompt(
            source_lang=src_lang,
            target_lang=self.target_lang,
            context_summary=context_summary,
            context_package=context_package # Pass full package for hard/soft glossary
        )

        # Construct user message
        chunk_text = context_package.get('chunk', '')
        user_message = f"""Translate the following chunk:

{chunk_text}
"""

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        return messages
        
    def _parse_llm_response(self, content: str, original_chunk: str) -> Dict[str, Any]:
        """Parse LLM response handling JSON or fallback to text"""
        content = content.strip()
        
        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON (starts with { and end with })
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # No JSON found, assume plain text translation
                return {
                    "translation": content,
                    "term_candidates": {},
                    "warnings": ["JSON parsing failed, used raw output"]
                }
                
        try:
            data = json.loads(json_str)
            return {
                "translation": data.get("translated_text", ""),
                "term_candidates": data.get("term_candidates", {}),
                "warnings": []
            }
        except json.JSONDecodeError:
             return {
                "translation": content, # Fallback to raw content if parse fails
                "term_candidates": {},
                "warnings": ["JSON decode error"]
            }

    def _get_system_prompt(self) -> str:
        """Get system prompt based on preset type with target language"""
        lang_names = {"ko": "Korean", "ja": "Japanese", "en": "English", "auto": "detected language"}
        target_name = lang_names.get(self.target_lang, self.target_lang)
        
        base_instruction = f"""You MUST translate into {target_name}. 
CRITICAL RULES:
- Output ONLY the translated text
- Translate the COMPLETE text, do not skip or summarize any part
- NEVER add '...' or ellipsis or any truncation markers
- Do NOT add explanations or notes
- Preserve all original content fully"""
        
        if self.preset_type == PresetType.SUBTITLE:
            return f"You are a professional subtitle translator. {base_instruction} Maintain natural spoken style and timing constraints."

        elif self.preset_type == PresetType.PATENT:
            return f"You are a specialized patent translator with legal and technical expertise. {base_instruction} Prioritize legal precision and technical accuracy."

        elif self.preset_type == PresetType.PAPER:
            return f"You are an expert academic translator specializing in research papers. {base_instruction} Use precise academic terminology and maintain formal tone."

        elif self.preset_type == PresetType.NOVEL:
            return f"You are a literary translator specializing in fiction. {base_instruction} Preserve author's voice and narrative flow."

        elif self.preset_type == PresetType.TECHNICAL:
            return f"You are a technical documentation translator. {base_instruction} Use clear, unambiguous language."

        else:  # GENERAL
            return f"You are a professional translator. {base_instruction} Produce natural, fluent translations that preserve meaning and intent."

    def _extract_term_candidates(self, source: str, target: str) -> List[Dict[str, Any]]:
        """
        Extract potential term candidates from translation.

        Simple heuristic: words in source that are different from target
        """
        candidates = []

        # Simple tokenization
        source_words = set(source.lower().split())
        target_words = set(target.lower().split())

        # Find words that changed
        changed_words = source_words - target_words

        for word in changed_words:
            if len(word) > 2:  # Skip short words
                candidates.append({
                    "source": word,
                    "target": word,  # Placeholder - would use dictionary/LLM in real implementation
                    "confidence": 0.7,
                    "context": source[:100]
                })

        return candidates

    def update_glossary_with_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Update glossary with term candidates.

        Returns: Dict with update stats
        """
        stats = {
            "added": 0,
            "skipped": 0,
            "total": len(candidates)
        }

        for candidate in candidates:
            # In real implementation, would validate and add
            stats["added"] += 1

        return stats

    def test_connection(self) -> bool:
        """Test LLM connection"""
        return self.llm_client.test_connection()

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            "preset_type": self.preset_type,
            "config": {
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
            }
        }
