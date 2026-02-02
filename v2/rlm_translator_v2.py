"""
RLM-Trans Main Translator Engine v2
With preset support for document type specific translations
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from config import LLMConfig, LANGUAGE_NAMES
from llm_client import LLMClient, LLMResponse
from repl_environment import TranslationREPL
from presets_v1 import TranslationPreset, PresetManager, get_preset_manager, LLMParameters
from text_utils import detect_language, chunk_text, clean_text, is_srt_format, parse_srt, format_srt


@dataclass
class TranslationResult:
    """Translation result with metadata"""
    translated_text: str
    source_lang: str
    target_lang: str
    chunks_count: int
    glossary: Dict[str, str]
    cost_summary: Dict[str, Any]
    preset_used: str = "general"
    success: bool = True
    error_message: str = ""


class RLMTranslatorV2:
    """
    RLM-based translator v2 with preset support.
    Supports document type specific translation settings.
    """
    
    def __init__(self, 
                 llm_config: Optional[LLMConfig] = None,
                 preset_name: str = "general",
                 progress_callback: Optional[callable] = None):
        """
        Initialize the RLM translator v2.
        
        Args:
            llm_config: LLM provider configuration
            preset_name: Name of preset to use (subtitle, paper, patent, novel, technical, general)
            progress_callback: Optional callback for progress updates
        """
        self.llm_config = llm_config or LLMConfig.from_env()
        self.progress_callback = progress_callback
        
        # Initialize preset manager
        self.preset_manager = get_preset_manager()
        self._current_preset_name = preset_name
        self._current_preset = self.preset_manager.get(preset_name)
        
        if not self._current_preset:
            self._current_preset = self.preset_manager.get("general")
            self._current_preset_name = "general"
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.llm_config)
        
        # REPL environment
        self.repl: Optional[TranslationREPL] = None
    
    @property
    def current_preset(self) -> TranslationPreset:
        return self._current_preset
    
    @property
    def current_preset_name(self) -> str:
        return self._current_preset_name
    
    def set_preset(self, preset_name: str) -> bool:
        """Change the current preset"""
        preset = self.preset_manager.get(preset_name)
        if preset:
            self._current_preset = preset
            self._current_preset_name = preset_name
            return True
        return False
    
    def list_presets(self) -> List[Dict[str, str]]:
        """List available presets with info"""
        return self.preset_manager.list_presets_with_info()
    
    def _report_progress(self, message: str, progress: float = 0.0):
        """Report progress to callback if set"""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def _get_llm_kwargs(self) -> Dict[str, Any]:
        """Get LLM parameters from current preset"""
        params = self._current_preset.llm_params
        return {
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
        }
    
    def _build_system_prompt(self, source_lang: str, target_lang: str, 
                              context_summary: str = "", glossary: Dict[str, str] = None) -> str:
        """Build system prompt from preset"""
        preset = self._current_preset
        
        # Format glossary
        if glossary:
            glossary_str = "\n".join([f"  - {k} → {v}" for k, v in glossary.items()])
        else:
            glossary_str = "  (No terms defined yet)"
        
        # Language names
        lang_names = {"ko": "Korean", "ja": "Japanese", "en": "English"}
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""{preset.system_prompt}

TRANSLATION CONTEXT:
- Source Language: {source_name}
- Target Language: {target_name}
- Previous context: {context_summary or "(Beginning of document)"}
- Style Guide: {preset.style_guide}

GLOSSARY (use these translations consistently):
{glossary_str}

{preset.context_instructions}
"""
        return prompt
    
    def translate(self, text: str, 
                  source_lang: Optional[str] = None,
                  target_lang: Optional[str] = None) -> TranslationResult:
        """
        Translate the given text using current preset.
        
        Args:
            text: Source text to translate
            source_lang: Source language (auto-detect if None)
            target_lang: Target language
            
        Returns:
            TranslationResult with translated text and metadata
        """
        # Clean input text
        text = clean_text(text)
        
        # Detect source language if not specified
        if not source_lang or source_lang == "auto":
            source_lang = detect_language(text)
            self._report_progress(f"Detected language: {LANGUAGE_NAMES.get(source_lang, source_lang)}")
        
        target_lang = target_lang or "ko"
        preset = self._current_preset
        
        self._report_progress(
            f"[{preset.name}] {LANGUAGE_NAMES.get(source_lang, source_lang)} → {LANGUAGE_NAMES.get(target_lang, target_lang)}"
        )
        
        # Check for SRT format
        if is_srt_format(text) and preset.document_type == "subtitle":
            return self._translate_srt(text, source_lang, target_lang)
        
        # Check text length
        if len(text) <= preset.chunk_size:
            return self._translate_simple(text, source_lang, target_lang)
        else:
            return self._translate_rlm(text, source_lang, target_lang)
    
    def _translate_simple(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Simple direct translation for short texts"""
        self._report_progress("Using direct translation (short text)")
        
        system_prompt = self._build_system_prompt(source_lang, target_lang)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            response = self.llm_client.complete(
                messages, 
                is_sub_call=True,
                **self._get_llm_kwargs()
            )
            
            return TranslationResult(
                translated_text=response.content.strip(),
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=1,
                glossary={},
                cost_summary=self.llm_client.cost_summary(),
                preset_used=self._current_preset_name
            )
        except Exception as e:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=0,
                glossary={},
                cost_summary=self.llm_client.cost_summary(),
                preset_used=self._current_preset_name,
                success=False,
                error_message=str(e)
            )
    
    def _translate_rlm(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Full RLM translation for long texts"""
        preset = self._current_preset
        self._report_progress(f"Using RLM translation (chunk size: {preset.chunk_size})", 0.0)
        
        # Create sub-agent query function
        def sub_agent_query(chunk: str) -> str:
            return self._call_sub_agent(chunk, source_lang, target_lang)
        
        # Initialize REPL environment
        self.repl = TranslationREPL(sub_agent_query)
        self.repl.set_original_text(text)
        
        # Prepare chunks using preset chunk_size
        chunks = chunk_text(text, preset.chunk_size)
        total_chunks = len(chunks)
        self._report_progress(f"Split into {total_chunks} chunks", 0.1)
        
        try:
            translated_chunks = []
            
            for i, (start, end, chunk) in enumerate(chunks):
                progress = 0.1 + (i / total_chunks) * 0.8
                self._report_progress(f"Translating chunk {i+1}/{total_chunks}", progress)
                
                # Build context
                context_summary = ""
                if translated_chunks:
                    last_chunk = translated_chunks[-1][-200:] if translated_chunks[-1] else ""
                    context_summary = f"Previous translation ended with: ...{last_chunk}"
                
                # Call sub-agent with preset settings
                translated = self._call_sub_agent(
                    chunk, source_lang, target_lang,
                    context_summary=context_summary,
                    glossary=self.repl.state.glossary
                )
                
                translated_chunks.append(translated)
                self.repl.state.translated_chunks.append(translated)
                self.repl.state.context_summary = f"Translated {i+1}/{total_chunks} chunks"
            
            final_text = ''.join(translated_chunks)
            self._report_progress("Translation complete", 1.0)
            
            return TranslationResult(
                translated_text=final_text,
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=total_chunks,
                glossary=self.repl.state.glossary,
                cost_summary=self.llm_client.cost_summary(),
                preset_used=self._current_preset_name
            )
            
        except Exception as e:
            return TranslationResult(
                translated_text=''.join(self.repl.state.translated_chunks) if self.repl else "",
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=len(self.repl.state.translated_chunks) if self.repl else 0,
                glossary=self.repl.state.glossary if self.repl else {},
                cost_summary=self.llm_client.cost_summary(),
                preset_used=self._current_preset_name,
                success=False,
                error_message=str(e)
            )
    
    def _call_sub_agent(self, chunk: str, source_lang: str, target_lang: str,
                        context_summary: str = "", glossary: Dict[str, str] = None) -> str:
        """Call sub-agent to translate a single chunk"""
        glossary = glossary or {}
        
        system_prompt = self._build_system_prompt(
            source_lang, target_lang, context_summary, glossary
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
        
        response = self.llm_client.complete(
            messages, 
            is_sub_call=True,
            **self._get_llm_kwargs()
        )
        return response.content.strip()
    
    def _translate_srt(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Translate SRT subtitle file"""
        self._report_progress("Detected SRT format, parsing subtitles")
        preset = self._current_preset
        
        entries = parse_srt(text)
        total = len(entries)
        
        if total == 0:
            return TranslationResult(
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=0,
                glossary={},
                cost_summary=self.llm_client.cost_summary(),
                preset_used=self._current_preset_name,
                success=False,
                error_message="Failed to parse SRT file"
            )
        
        self._report_progress(f"Translating {total} subtitle entries")
        
        # Initialize REPL
        def sub_query(chunk): return self._call_sub_agent(chunk, source_lang, target_lang)
        self.repl = TranslationREPL(sub_query)
        
        batch_size = 10
        translated_entries = []
        
        for i in range(0, total, batch_size):
            batch = entries[i:i+batch_size]
            progress = i / total
            self._report_progress(f"Subtitles {i+1}-{min(i+batch_size, total)}/{total}", progress)
            
            batch_text = '\n---\n'.join([e['text'] for e in batch])
            
            context = f"Subtitle lines {i+1} to {min(i+batch_size, total)} of {total}"
            translated = self._call_sub_agent(
                batch_text, source_lang, target_lang,
                context_summary=context, glossary=self.repl.state.glossary
            )
            
            translated_parts = translated.split('\n---\n')
            
            for j, entry in enumerate(batch):
                new_entry = entry.copy()
                if j < len(translated_parts):
                    new_entry['text'] = translated_parts[j].strip()
                translated_entries.append(new_entry)
        
        result_text = format_srt(translated_entries)
        self._report_progress("Translation complete", 1.0)
        
        return TranslationResult(
            translated_text=result_text,
            source_lang=source_lang,
            target_lang=target_lang,
            chunks_count=len(entries),
            glossary=self.repl.state.glossary if self.repl else {},
            cost_summary=self.llm_client.cost_summary(),
            preset_used=self._current_preset_name
        )
    
    def test_connection(self) -> bool:
        return self.llm_client.test_connection()
    
    def list_models(self) -> List[str]:
        return self.llm_client.list_models()
    
    def reset_costs(self):
        self.llm_client.reset_costs()
    
    # Preset management methods
    def save_current_preset_as(self, key: str, name: str) -> bool:
        """Save current preset with new name"""
        if self._current_preset:
            new_preset = TranslationPreset.from_dict(self._current_preset.to_dict())
            new_preset.name = name
            new_preset.created_at = ""
            new_preset.modified_at = ""
            self.preset_manager.save_preset(key, new_preset)
            return True
        return False
    
    def update_preset_llm_params(self, **kwargs) -> None:
        """Update current preset's LLM parameters"""
        if self._current_preset:
            for key, value in kwargs.items():
                if hasattr(self._current_preset.llm_params, key):
                    setattr(self._current_preset.llm_params, key, value)
    
    def get_preset_info(self) -> Dict[str, Any]:
        """Get current preset info for display"""
        if not self._current_preset:
            return {}
        
        return {
            "name": self._current_preset.name,
            "description": self._current_preset.description,
            "document_type": self._current_preset.document_type,
            "temperature": self._current_preset.llm_params.temperature,
            "max_tokens": self._current_preset.llm_params.max_tokens,
            "top_p": self._current_preset.llm_params.top_p,
            "chunk_size": self._current_preset.chunk_size,
            "style_guide": self._current_preset.style_guide,
        }


# CLI test
if __name__ == "__main__":
    import sys
    
    print("RLM Translator v2 - Preset System Test")
    print("-" * 50)
    
    translator = RLMTranslatorV2(
        preset_name="subtitle",
        progress_callback=lambda msg, prog: print(f"[{prog*100:.0f}%] {msg}")
    )
    
    print("\nAvailable Presets:")
    for preset in translator.list_presets():
        print(f"  - {preset['key']}: {preset['name']}")
    
    print(f"\nCurrent Preset: {translator.current_preset_name}")
    print(f"Preset Info: {translator.get_preset_info()}")
    
    if translator.test_connection():
        print("\n✓ LLM Connected")
        
        test_text = "안녕하세요. 오늘 정말 좋은 날씨네요."
        print(f"\nTest translation: {test_text}")
        
        result = translator.translate(test_text, target_lang="en")
        if result.success:
            print(f"Result: {result.translated_text}")
        else:
            print(f"Error: {result.error_message}")
    else:
        print("\n✗ Cannot connect to LLM server")
