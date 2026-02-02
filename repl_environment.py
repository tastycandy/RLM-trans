"""
RLM-Trans REPL Environment
Context storage and safe code execution for translation
"""
import sys
import io
import traceback
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field


@dataclass
class TranslationState:
    """Translation context stored in REPL environment"""
    original_text: str = ""
    translated_chunks: List[str] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    context_summary: str = ""
    chunk_plan: List[tuple] = field(default_factory=list)  # [(start, end), ...]
    current_chunk_index: int = 0
    

class TranslationREPL:
    """
    REPL environment for RLM translation.
    Stores translation context as external variables and provides
    safe execution of Python code from the LLM.
    """
    
    def __init__(self, llm_query_func: Callable[[str], str]):
        """
        Initialize REPL with LLM query function.
        
        Args:
            llm_query_func: Function to call sub-agent for chunk translation
        """
        self.state = TranslationState()
        self._llm_query_func = llm_query_func
        self._final_result: Optional[str] = None
        self._is_finished = False
        self._output_buffer = io.StringIO()
        
        # Build execution namespace
        self._namespace = self._build_namespace()
    
    def _build_namespace(self) -> Dict[str, Any]:
        """Build the namespace for code execution"""
        return {
            # State variables (will be updated dynamically)
            'original_text': self.state.original_text,
            'translated_chunks': self.state.translated_chunks,
            'glossary': self.state.glossary,
            'context_summary': self.state.context_summary,
            
            # Helper functions
            'llm_query': self._llm_query,
            'get_chunk': self._get_chunk,
            'update_glossary': self._update_glossary,
            'set_context_summary': self._set_context_summary,
            'add_translated_chunk': self._add_translated_chunk,
            'FINAL': self._final,
            'FINAL_VAR': self._final_var,
            
            # Utilities
            'print': self._safe_print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
        }
    
    def set_original_text(self, text: str):
        """Set the original text to translate"""
        self.state.original_text = text
        self._namespace['original_text'] = text
    
    def _llm_query(self, prompt: str) -> str:
        """Call sub-agent for translation"""
        return self._llm_query_func(prompt)
    
    def _get_chunk(self, start: int, end: int) -> str:
        """Get a portion of the original text"""
        return self.state.original_text[start:end]
    
    def _update_glossary(self, source: str, target: str):
        """Add or update a glossary entry"""
        self.state.glossary[source] = target
        self._namespace['glossary'] = self.state.glossary
        self._safe_print(f"Glossary updated: {source} → {target}")
    
    def _set_context_summary(self, summary: str):
        """Update the context summary"""
        self.state.context_summary = summary
        self._namespace['context_summary'] = summary
    
    def _add_translated_chunk(self, chunk: str):
        """Add a translated chunk to the list"""
        self.state.translated_chunks.append(chunk)
        self._namespace['translated_chunks'] = self.state.translated_chunks
        self._safe_print(f"Added chunk {len(self.state.translated_chunks)}")
    
    def _final(self, result: str):
        """Set the final result and mark as finished"""
        self._final_result = result
        self._is_finished = True
        self._safe_print("FINAL result set.")
    
    def _final_var(self, var_name: str):
        """Use a variable's value as the final result"""
        if var_name in self._namespace:
            value = self._namespace[var_name]
            if isinstance(value, list):
                self._final_result = ''.join(str(v) for v in value)
            else:
                self._final_result = str(value)
            self._is_finished = True
            self._safe_print(f"FINAL_VAR: Using {var_name} as result")
        else:
            self._safe_print(f"Error: Variable '{var_name}' not found")
    
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
        self._namespace['original_text'] = self.state.original_text
        self._namespace['translated_chunks'] = self.state.translated_chunks
        self._namespace['glossary'] = self.state.glossary
        self._namespace['context_summary'] = self.state.context_summary
        
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
        import re
        
        # Try to find ```python ... ``` blocks
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return '\n'.join(matches)
        
        # If no code blocks, assume the entire text is code
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
            'text_length': len(self.state.original_text),
            'chunks_translated': len(self.state.translated_chunks),
            'glossary_size': len(self.state.glossary),
            'context_summary': self.state.context_summary[:500] if self.state.context_summary else "",
        }
    
    def reset(self):
        """Reset the REPL environment"""
        self.state = TranslationState()
        self._final_result = None
        self._is_finished = False
        self._namespace = self._build_namespace()
