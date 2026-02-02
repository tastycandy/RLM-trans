"""
RLM-Trans Configuration Manager
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class LLMConfig:
    """LLM Provider Configuration"""
    provider: str = "lmstudio"  # lmstudio, openai, gemini
    
    # LM Studio
    lm_studio_url: str = "http://100.120.52.113:25435/v1"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # Gemini
    gemini_api_key: Optional[str] = None
    
    # Models
    root_model: Optional[str] = None  # Main agent model
    sub_model: Optional[str] = None   # Sub agent model (for chunk translation)
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables"""
        return cls(
            provider=os.getenv("DEFAULT_PROVIDER", "lmstudio"),
            lm_studio_url=os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            root_model=os.getenv("ROOT_MODEL"),
            sub_model=os.getenv("SUB_MODEL"),
        )


@dataclass
class TranslationConfig:
    """Translation Settings"""
    source_lang: str = "auto"  # auto, ko, ja, en
    target_lang: str = "ko"    # ko, ja, en
    
    # RLM Settings
    max_iterations: int = 20
    max_output_length: int = 500000
    chunk_size: int = 2000  # Characters per chunk
    
    # Translation Quality
    preserve_formatting: bool = True
    use_glossary: bool = True


LANGUAGE_NAMES = {
    "ko": "Korean",
    "ja": "Japanese", 
    "en": "English",
    "auto": "Auto-detect"
}

LANGUAGE_NAMES_KO = {
    "ko": "한국어",
    "ja": "일본어",
    "en": "영어",
    "auto": "자동 감지"
}
