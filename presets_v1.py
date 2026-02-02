"""
RLM-Trans Presets System v1
Document type specific presets with LLM parameters and prompts
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class LLMParameters:
    """LLM generation parameters"""
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass 
class TranslationPreset:
    """Complete translation preset configuration"""
    # Metadata
    name: str
    description: str
    document_type: str  # subtitle, paper, patent, novel, general, etc.
    version: int = 1
    created_at: str = ""
    modified_at: str = ""
    
    # LLM Parameters
    llm_params: LLMParameters = field(default_factory=LLMParameters)
    
    # Translation Settings
    chunk_size: int = 2000
    preserve_formatting: bool = True
    use_glossary: bool = True
    
    # Custom Prompts
    system_prompt: str = ""
    context_instructions: str = ""
    style_guide: str = ""
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.modified_at:
            self.modified_at = now
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslationPreset":
        """Create from dictionary"""
        # Handle nested LLMParameters
        if "llm_params" in data and isinstance(data["llm_params"], dict):
            data["llm_params"] = LLMParameters(**data["llm_params"])
        return cls(**data)
    
    def update_modified(self):
        """Update modification timestamp"""
        self.modified_at = datetime.now().isoformat()


# Default presets for different document types
DEFAULT_PRESETS = {
    "subtitle": TranslationPreset(
        name="자막 (Subtitle)",
        description="영상 자막 번역에 최적화. 짧고 자연스러운 표현 사용.",
        document_type="subtitle",
        llm_params=LLMParameters(
            temperature=0.3,
            max_tokens=2048,
            top_p=0.9
        ),
        chunk_size=1500,
        preserve_formatting=True,
        use_glossary=True,
        system_prompt="""You are a professional subtitle translator.
        
RULES:
- Keep translations SHORT and natural for spoken dialogue
- Match the timing constraints of subtitles
- Preserve speaker's tone and emotion
- Use colloquial expressions appropriate for speech
- Avoid overly formal or literary language
- Keep line breaks where they make sense for readability""",
        context_instructions="Consider natural speech patterns and subtitle timing.",
        style_guide="자연스러운 구어체, 간결한 표현"
    ),
    
    "paper": TranslationPreset(
        name="논문 (Academic Paper)",
        description="학술 논문 번역에 최적화. 정확하고 학술적인 표현 사용.",
        document_type="paper",
        llm_params=LLMParameters(
            temperature=0.2,
            max_tokens=4096,
            top_p=0.85
        ),
        chunk_size=2500,
        preserve_formatting=True,
        use_glossary=True,
        system_prompt="""You are an expert academic translator specializing in research papers.

RULES:
- Use precise academic terminology
- Maintain formal, objective tone
- Preserve technical terms (transliterate if no standard translation exists)
- Keep citation formats intact
- Translate figure/table captions accurately
- Maintain logical flow and argumentation structure""",
        context_instructions="Preserve academic rigor and citation formats.",
        style_guide="학술적 문체, 전문 용어 유지"
    ),
    
    "patent": TranslationPreset(
        name="특허 (Patent)",
        description="특허 문서 번역에 최적화. 법적 정확성과 기술 용어 중시.",
        document_type="patent",
        llm_params=LLMParameters(
            temperature=0.1,  # Very low for precision
            max_tokens=4096,
            top_p=0.8
        ),
        chunk_size=2000,
        preserve_formatting=True,
        use_glossary=True,
        system_prompt="""You are a specialized patent translator with legal and technical expertise.

RULES:
- Use EXACT legal terminology - precision is critical
- Maintain claim structure and numbering
- Preserve all technical specifications exactly
- Keep patent-specific phrases (e.g., "comprising", "wherein")
- Do not paraphrase - translate as literally as legally appropriate
- Maintain reference numbers and figure references""",
        context_instructions="Legal precision is paramount. Technical terms must be consistent.",
        style_guide="법적 정확성, 기술 용어 일관성"
    ),
    
    "novel": TranslationPreset(
        name="소설 (Novel/Fiction)",
        description="소설 및 문학 작품 번역. 문체와 감성 보존.",
        document_type="novel",
        llm_params=LLMParameters(
            temperature=0.5,  # Higher for creative expression
            max_tokens=4096,
            top_p=0.95
        ),
        chunk_size=3000,
        preserve_formatting=True,
        use_glossary=True,
        system_prompt="""You are a literary translator specializing in fiction.

RULES:
- Preserve the author's unique voice and style
- Maintain narrative flow and pacing
- Translate idioms naturally, not literally
- Keep character voice distinctions
- Preserve metaphors and literary devices when possible
- Adapt cultural references appropriately
- Maintain emotional impact and atmosphere""",
        context_instructions="Focus on literary quality and emotional resonance.",
        style_guide="문학적 표현, 작가 스타일 보존"
    ),
    
    "technical": TranslationPreset(
        name="기술 문서 (Technical)",
        description="기술 문서, 매뉴얼 번역. 명확성과 정확성 중시.",
        document_type="technical",
        llm_params=LLMParameters(
            temperature=0.2,
            max_tokens=4096,
            top_p=0.85
        ),
        chunk_size=2000,
        preserve_formatting=True,
        use_glossary=True,
        system_prompt="""You are a technical documentation translator.

RULES:
- Use clear, unambiguous language
- Maintain consistent terminology throughout
- Preserve code snippets and commands exactly
- Keep formatting (lists, headings, tables)
- Translate UI text according to localization standards
- Keep placeholder text and variables unchanged""",
        context_instructions="Clarity and consistency are essential.",
        style_guide="명확한 표현, 일관된 용어"
    ),
    
    "general": TranslationPreset(
        name="일반 (General)",
        description="범용 번역 설정. 다양한 문서에 적합.",
        document_type="general",
        llm_params=LLMParameters(
            temperature=0.3,
            max_tokens=4096,
            top_p=0.9
        ),
        chunk_size=2000,
        preserve_formatting=True,
        use_glossary=True,
        system_prompt="""You are a professional translator.

RULES:
- Produce natural, fluent translations
- Preserve the meaning and intent of the original
- Maintain appropriate formality level
- Keep formatting and structure intact""",
        context_instructions="Balance accuracy with natural expression.",
        style_guide="자연스러운 번역"
    )
}


class PresetManager:
    """Manage translation presets - load, save, modify"""
    
    def __init__(self, presets_dir: Optional[Path] = None):
        self.presets_dir = presets_dir or Path(__file__).parent / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        self._presets: Dict[str, TranslationPreset] = {}
        
        # Initialize with defaults then load custom
        self._load_defaults()
        self._load_custom_presets()
    
    def _load_defaults(self):
        """Load default presets"""
        for key, preset in DEFAULT_PRESETS.items():
            self._presets[key] = preset
    
    def _load_custom_presets(self):
        """Load custom presets from JSON files"""
        for json_file in self.presets_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    preset = TranslationPreset.from_dict(data)
                    key = json_file.stem
                    self._presets[key] = preset
            except Exception as e:
                print(f"Error loading preset {json_file}: {e}")
    
    def get(self, name: str) -> Optional[TranslationPreset]:
        """Get preset by name"""
        return self._presets.get(name)
    
    def list_presets(self) -> List[str]:
        """List all preset names"""
        return list(self._presets.keys())
    
    def list_presets_with_info(self) -> List[Dict[str, str]]:
        """List presets with display info"""
        return [
            {
                "key": key,
                "name": preset.name,
                "description": preset.description,
                "document_type": preset.document_type
            }
            for key, preset in self._presets.items()
        ]
    
    def save_preset(self, key: str, preset: TranslationPreset) -> Path:
        """Save preset to JSON file"""
        preset.update_modified()
        file_path = self.presets_dir / f"{key}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)
        
        self._presets[key] = preset
        return file_path
    
    def create_custom_preset(self, key: str, name: str, 
                             base_preset: str = "general",
                             **overrides) -> TranslationPreset:
        """Create new preset based on existing one"""
        base = self.get(base_preset) or DEFAULT_PRESETS["general"]
        
        # Copy base preset data
        data = base.to_dict()
        data["name"] = name
        data["created_at"] = ""
        data["modified_at"] = ""
        data["version"] = 1
        
        # Apply overrides
        for k, v in overrides.items():
            if k == "llm_params" and isinstance(v, dict):
                for param_key, param_val in v.items():
                    data["llm_params"][param_key] = param_val
            else:
                data[k] = v
        
        preset = TranslationPreset.from_dict(data)
        self.save_preset(key, preset)
        
        return preset
    
    def delete_preset(self, key: str) -> bool:
        """Delete custom preset (cannot delete defaults)"""
        if key in DEFAULT_PRESETS:
            return False  # Cannot delete defaults
        
        file_path = self.presets_dir / f"{key}.json"
        if file_path.exists():
            file_path.unlink()
        
        if key in self._presets:
            del self._presets[key]
        
        return True
    
    def export_preset(self, key: str, file_path: Path) -> bool:
        """Export preset to external file"""
        preset = self.get(key)
        if not preset:
            return False
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)
        
        return True
    
    def import_preset(self, file_path: Path, key: Optional[str] = None) -> Optional[TranslationPreset]:
        """Import preset from external file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preset = TranslationPreset.from_dict(data)
            
            if key is None:
                key = file_path.stem
            
            self.save_preset(key, preset)
            return preset
        except Exception as e:
            print(f"Error importing preset: {e}")
            return None


# Singleton instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """Get global preset manager instance"""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager
