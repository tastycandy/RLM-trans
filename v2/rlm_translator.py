"""
RLM-Trans Main Translator Engine
Recursive Language Model based translation with context preservation
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from config import LLMConfig, TranslationConfig, LANGUAGE_NAMES
from llm_client import LLMClient, LLMResponse
from repl_environment import TranslationREPL
from prompts import ROOT_AGENT_SYSTEM, get_sub_agent_prompt
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
    success: bool = True
    error_message: str = ""


class RLMTranslator:
    """
    Main RLM-based translator class.
    Uses a root agent to coordinate translation and sub-agents for chunk translation.
    """
    
    def __init__(self, 
                 llm_config: Optional[LLMConfig] = None,
                 translation_config: Optional[TranslationConfig] = None,
                 progress_callback: Optional[callable] = None):
        """
        Initialize the RLM translator.
        
        Args:
            llm_config: LLM provider configuration
            translation_config: Translation settings
            progress_callback: Optional callback for progress updates (message: str, progress: float)
        """
        self.llm_config = llm_config or LLMConfig.from_env()
        self.trans_config = translation_config or TranslationConfig()
        self.progress_callback = progress_callback
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.llm_config)
        
        # REPL environment (created per translation)
        self.repl: Optional[TranslationREPL] = None
        
    def _report_progress(self, message: str, progress: float = 0.0):
        """Report progress to callback if set"""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def translate(self, text: str, 
                  source_lang: Optional[str] = None,
                  target_lang: Optional[str] = None) -> TranslationResult:
        """
        Translate the given text.
        
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
        
        target_lang = target_lang or self.trans_config.target_lang
        
        self._report_progress(f"Translating {LANGUAGE_NAMES.get(source_lang, source_lang)} → {LANGUAGE_NAMES.get(target_lang, target_lang)}")
        
        # Check for SRT format
        if is_srt_format(text):
            return self._translate_srt(text, source_lang, target_lang)
        
        # Check text length
        if len(text) <= self.trans_config.chunk_size:
            # Short text: direct translation without RLM
            return self._translate_simple(text, source_lang, target_lang)
        else:
            # Long text: use RLM approach
            return self._translate_rlm(text, source_lang, target_lang)
    
    def _translate_simple(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Simple direct translation for short texts"""
        self._report_progress("Using direct translation (short text)")
        
        prompt = get_sub_agent_prompt(source_lang, target_lang, "", {})
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            response = self.llm_client.complete(messages, is_sub_call=True)
            
            return TranslationResult(
                translated_text=response.content.strip(),
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=1,
                glossary={},
                cost_summary=self.llm_client.cost_summary()
            )
        except Exception as e:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=0,
                glossary={},
                cost_summary=self.llm_client.cost_summary(),
                success=False,
                error_message=str(e)
            )
    
    def _translate_rlm(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Full RLM translation for long texts"""
        self._report_progress("Using RLM translation (long text)", 0.0)
        
        # Create sub-agent query function
        def sub_agent_query(chunk: str) -> str:
            return self._call_sub_agent(chunk, source_lang, target_lang)
        
        # Initialize REPL environment
        self.repl = TranslationREPL(sub_agent_query)
        self.repl.set_original_text(text)
        
        # Prepare chunks
        chunks = chunk_text(text, self.trans_config.chunk_size)
        total_chunks = len(chunks)
        self._report_progress(f"Split into {total_chunks} chunks", 0.1)
        
        # Prepare initial glossary (extract potential terms)
        from text_utils import extract_terms
        potential_terms = extract_terms(text)
        
        try:
            # Strategy 1: Simplified iterative translation
            # (Full RLM with root agent code execution could be added later)
            translated_chunks = []
            
            for i, (start, end, chunk) in enumerate(chunks):
                progress = 0.1 + (i / total_chunks) * 0.8
                self._report_progress(f"Translating chunk {i+1}/{total_chunks}", progress)
                
                # Build context for this chunk
                context_summary = ""
                if translated_chunks:
                    # Summarize previous translations for context
                    last_chunk = translated_chunks[-1][-200:] if translated_chunks[-1] else ""
                    context_summary = f"Previous translation ended with: ...{last_chunk}"
                
                # Call sub-agent
                translated = self._call_sub_agent(
                    chunk, 
                    source_lang, 
                    target_lang,
                    context_summary=context_summary,
                    glossary=self.repl.state.glossary
                )
                
                translated_chunks.append(translated)
                self.repl.state.translated_chunks.append(translated)
                
                # Update context summary
                self.repl.state.context_summary = f"Translated {i+1}/{total_chunks} chunks"
            
            # Combine all chunks
            final_text = ''.join(translated_chunks)
            
            self._report_progress("Translation complete", 1.0)
            
            return TranslationResult(
                translated_text=final_text,
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=total_chunks,
                glossary=self.repl.state.glossary,
                cost_summary=self.llm_client.cost_summary()
            )
            
        except Exception as e:
            return TranslationResult(
                translated_text=''.join(self.repl.state.translated_chunks),
                source_lang=source_lang,
                target_lang=target_lang,
                chunks_count=len(self.repl.state.translated_chunks),
                glossary=self.repl.state.glossary,
                cost_summary=self.llm_client.cost_summary(),
                success=False,
                error_message=str(e)
            )
    
    def _call_sub_agent(self, chunk: str, source_lang: str, target_lang: str,
                        context_summary: str = "", glossary: Dict[str, str] = None) -> str:
        """Call sub-agent to translate a single chunk"""
        glossary = glossary or {}
        
        prompt = get_sub_agent_prompt(source_lang, target_lang, context_summary, glossary)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk}
        ]
        
        response = self.llm_client.complete(messages, is_sub_call=True)
        return response.content.strip()
    
    def _translate_srt(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Translate SRT subtitle file"""
        self._report_progress("Detected SRT format, parsing subtitles")
        
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
                success=False,
                error_message="Failed to parse SRT file"
            )
        
        self._report_progress(f"Translating {total} subtitle entries")
        
        # Create sub-agent query function
        def sub_agent_query(chunk: str) -> str:
            return self._call_sub_agent(chunk, source_lang, target_lang)
        
        self.repl = TranslationREPL(sub_agent_query)
        
        # Batch subtitle entries for efficiency
        batch_size = 10
        translated_entries = []
        
        for i in range(0, total, batch_size):
            batch = entries[i:i+batch_size]
            progress = i / total
            self._report_progress(f"Translating subtitles {i+1}-{min(i+batch_size, total)}/{total}", progress)
            
            # Combine batch texts
            batch_text = '\n---\n'.join([e['text'] for e in batch])
            
            # Translate batch
            context = f"Subtitle lines {i+1} to {min(i+batch_size, total)} of {total}"
            translated = self._call_sub_agent(
                batch_text, source_lang, target_lang,
                context_summary=context, glossary=self.repl.state.glossary
            )
            
            # Split translated text back
            translated_parts = translated.split('\n---\n')
            
            for j, entry in enumerate(batch):
                new_entry = entry.copy()
                if j < len(translated_parts):
                    new_entry['text'] = translated_parts[j].strip()
                translated_entries.append(new_entry)
        
        # Format back to SRT
        result_text = format_srt(translated_entries)
        
        self._report_progress("Translation complete", 1.0)
        
        return TranslationResult(
            translated_text=result_text,
            source_lang=source_lang,
            target_lang=target_lang,
            chunks_count=len(entries),
            glossary=self.repl.state.glossary if self.repl else {},
            cost_summary=self.llm_client.cost_summary()
        )
    
    def test_connection(self) -> bool:
        """Test LLM connection"""
        return self.llm_client.test_connection()
    
    def list_models(self) -> List[str]:
        """List available models"""
        return self.llm_client.list_models()
    
    def reset_costs(self):
        """Reset cost tracking"""
        self.llm_client.reset_costs()


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    print("RLM Translator - Command Line Test")
    print("-" * 40)
    
    # Create translator
    translator = RLMTranslator(
        progress_callback=lambda msg, prog: print(f"[{prog*100:.0f}%] {msg}")
    )
    
    # Test connection
    if not translator.test_connection():
        print("Error: Cannot connect to LLM server")
        print("Make sure LM Studio is running or API keys are configured in .env")
        sys.exit(1)
    
    print("Connected to LLM server")
    print(f"Available models: {translator.list_models()}")
    
    # Test translation
    test_text = """
    안녕하세요. 오늘 날씨가 정말 좋네요.
    저는 인공지능 번역 시스템을 개발하고 있습니다.
    이 시스템은 재귀 언어 모델을 사용하여 긴 문서도 번역할 수 있습니다.
    """
    
    print("\nSource text:")
    print(test_text)
    
    result = translator.translate(test_text, target_lang="en")
    
    if result.success:
        print("\nTranslated text:")
        print(result.translated_text)
        print(f"\nCost summary: {result.cost_summary}")
    else:
        print(f"\nTranslation failed: {result.error_message}")
