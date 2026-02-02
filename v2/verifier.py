"""
RLM Verifier/Critic
Rule-based validation with optional LLM validation
"""
from typing import Dict, Any, List, Optional
from enum import Enum

from rlm_state import QualityFlagType, RepairType


class ValidationType(str, Enum):
    """Types of validation"""
    FORMAT = "format"
    COMPLETION = "completion"
    FORBIDDEN = "forbidden"
    TERMINOLOGY = "terminology"
    TONE = "tone"
    STRUCTURE = "structure"


class ErrorSeverity(str, Enum):
    """Severity levels for errors"""
    HARD = "hard"  # Must fix - format violation
    SOFT = "soft"  # Should fix - quality issue


class ValidationResult:
    """Result of validation"""

    def __init__(self, valid: bool = True):
        self.valid = valid
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.repair_type: Optional[RepairType] = None
        self.repair_description: Optional[str] = None

    def add_error(self, error_type, message: str, severity: ErrorSeverity = ErrorSeverity.HARD):
        """Add an error"""
        # Handle enum types
        if hasattr(error_type, 'value'):
            error_type = error_type.value
        self.errors.append({
            "type": error_type,
            "message": message,
            "severity": severity.value
        })
        self.valid = False

    def add_warning(self, warning_type, message: str):
        """Add a warning"""
        # Handle enum types
        if hasattr(warning_type, 'value'):
            warning_type = warning_type.value
        self.warnings.append({
            "type": warning_type,
            "message": message
        })

    def set_repair(self, repair_type: RepairType, description: str):
        """Set recommended repair action"""
        self.repair_type = repair_type
        self.repair_description = description

    def is_hard_error(self) -> bool:
        """Check if any hard errors exist"""
        return any(e["severity"] == ErrorSeverity.HARD.value for e in self.errors)

    def get_hard_error_types(self) -> List[str]:
        """Get list of hard error types"""
        return [e["type"] for e in self.errors if e["severity"] == ErrorSeverity.HARD.value]

    def summary(self) -> str:
        """Get validation summary"""
        if self.valid:
            return "Translation passed all validations"

        summary_lines = [
            f"Valid: {self.valid}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]

        if self.repair_type:
            summary_lines.append(f"Recommended repair: {self.repair_type.value}")

        return "\n".join(summary_lines)


class Verifier:
    """
    Verifier/Critic for translation validation.
    Performs rule-based validation and optional LLM validation.
    """

    def __init__(self, enable_llm_validation: bool = False):
        """
        Initialize verifier.

        Args:
            enable_llm_validation: Whether to enable LLM validation
        """
        self.enable_llm_validation = enable_llm_validation

    def validate(
        self,
        translation: str,
        original_chunk: str,
        context: Dict[str, Any],
        preset_type: str = "general",
        check_sentence: bool = True,
        check_length: bool = True
    ) -> ValidationResult:
        """
        Validate translation.

        Args:
            translation: Translated text
            original_chunk: Original chunk text
            context: Translation context
            preset_type: Document type preset
            check_sentence: Whether to check sentence completion
            check_length: Whether to check translation length

        Returns:
            ValidationResult with validation results and recommendations
        """
        result = ValidationResult()

        # Perform rule-based validation
        self._rule_based_validation(
            result, translation, original_chunk, context, preset_type,
            check_sentence=check_sentence, check_length=check_length
        )

        # Optionally perform LLM validation
        if not result.valid and self.enable_llm_validation:
            self._llm_validation(result, translation, original_chunk, context, preset_type)

        # Determine repair recommendation
        if not result.valid:
            self._determine_repair(result)

        return result

    def _rule_based_validation(
        self,
        result: ValidationResult,
        translation: str,
        original_chunk: str,
        context: Dict[str, Any],
        preset_type: str,
        check_sentence: bool = True,
        check_length: bool = True
    ):
        """Perform rule-based validation"""

        # Check for empty translation
        if not translation or translation.strip() == "":
            result.add_error(
                ValidationType.COMPLETION,
                "Translation is empty",
                ErrorSeverity.HARD
            )
            return

        # Check for truncation - ends with "..."
        if translation.rstrip().endswith("...") or translation.rstrip().endswith("…"):
            result.add_error(
                ValidationType.COMPLETION,
                "Translation appears truncated (ends with '...')",
                ErrorSeverity.HARD
            )
        
        # Check for sentence completion (문장 단위 검토)
        if check_sentence:
            # Check if translation ends with proper sentence-ending punctuation
            sentence_endings = ('.', '!', '?', '。', '！', '？', '다.', '요.', '니다.')
            trans_stripped = translation.rstrip()
            
            # Check if any sentence ending matches
            ends_with_sentence = any(trans_stripped.endswith(ending) for ending in sentence_endings)
            
            if not ends_with_sentence and len(trans_stripped) > 50:
                result.add_error(
                    ValidationType.COMPLETION,
                    "Translation does not end with complete sentence",
                    ErrorSeverity.HARD
                )
        
        # Check for length (길이 검토)
        if check_length:
            orig_len = len(original_chunk.strip())
            trans_len = len(translation.strip())
            
            # Translation should be at least 50% of original length
            if orig_len > 100 and trans_len < orig_len * 0.5:
                result.add_error(
                    ValidationType.COMPLETION,
                    f"Translation too short ({trans_len} chars vs original {orig_len} chars, <50%)",
                    ErrorSeverity.HARD
                )

        # Check format-specific rules based on preset
        if preset_type == "subtitle":
            self._validate_subtitle_format(result, translation, context)
        elif preset_type == "patent":
            self._validate_patent_format(result, translation, context)
        elif preset_type == "paper":
            self._validate_paper_format(result, translation, context)

        # Check for forbidden words/phrases
        self._check_forbidden_content(result, translation, context)

        # Check length constraints
        self._check_length_constraints(result, translation, original_chunk)

        # Check terminology consistency
        self._check_terminology(result, translation, context)

    def _validate_subtitle_format(self, result: ValidationResult, translation: str, context: Dict[str, Any]):
        """Validate subtitle-specific format rules"""
        # Check for empty lines (shouldn't happen in proper subtitle format)
        lines = translation.strip().split('\n')
        if len(lines) == 0:
            result.add_error(
                ValidationType.FORMAT,
                "Subtitle has no lines",
                ErrorSeverity.HARD
            )

    def _validate_patent_format(self, result: ValidationResult, translation: str, context: Dict[str, Any]):
        """Validate patent-specific format rules"""
        # Check for missing claim numbers
        if not any(word.isdigit() for word in translation.split()):
            result.add_warning(
                ValidationType.STRUCTURE,
                "No claim numbers found (typical in patent translations)"
            )

        # Check for proper clause markers
        if "wherein" not in translation.lower():
            result.add_warning(
                ValidationType.STRUCTURE,
                "Missing 'wherein' clause marker (optional)"
            )

    def _validate_paper_format(self, result: ValidationResult, translation: str, context: Dict[str, Any]):
        """Validate paper-specific format rules"""
        # Academic papers should have proper structure
        if translation.count('.') < 3:
            result.add_warning(
                ValidationType.STRUCTURE,
                "Paper may lack sufficient sentence structure"
            )

    def _check_forbidden_content(self, result: ValidationResult, translation: str, context: Dict[str, Any]):
        """Check for forbidden words or phrases"""
        forbidden_words = context.get("style", {}).get("forbidden_words", [])

        for word in forbidden_words:
            if word.lower() in translation.lower():
                result.add_error(
                    ValidationType.FORBIDDEN,
                    f"Contains forbidden word: '{word}'",
                    ErrorSeverity.HARD
                )

    def _check_length_constraints(self, result: ValidationResult, translation: str, original: str):
        """Check if translation length is reasonable"""
        orig_len = len(original)
        trans_len = len(translation)

        # Avoid extreme length changes (> 3x)
        if trans_len > orig_len * 3:
            result.add_warning(
                ValidationType.COMPLETION,
                "Translation is significantly longer than original (>3x)"
            )

        # Avoid empty translation
        if trans_len == 0:
            result.add_error(
                ValidationType.COMPLETION,
                "Translation is empty despite original having content",
                ErrorSeverity.HARD
            )

    def _check_terminology(self, result: ValidationResult, translation: str, context: Dict[str, Any]):
        """Check terminology consistency"""
        glossary = context.get("glossary", {})

        if glossary:
            # Check if glossary terms are used consistently
            glossary_sources = list(glossary.keys())

            # This is a simple check - would need more sophisticated analysis in real implementation
            for term in glossary_sources[:10]:  # Check first 10 terms
                term_lower = term.lower()
                count = translation.lower().count(term_lower)

                if count == 0:
                    result.add_warning(
                        ValidationType.TERMINOLOGY,
                        f"Glossary term '{term}' not found in translation"
                    )

    def _llm_validation(
        self,
        result: ValidationResult,
        translation: str,
        original_chunk: str,
        context: Dict[str, Any],
        preset_type: str
    ):
        """Perform LLM validation for quality issues"""
        # In real implementation, would call LLM for quality checks
        # Examples:
        # - Semantic preservation check
        # - Naturalness check
        # - Tone check
        # - Context understanding

        # Placeholder for LLM validation
        if not result.valid and not result.warnings:
            result.add_warning(
                ValidationType.TONE,
                "No specific quality issues detected, but could benefit from LLM validation"
            )

    def _determine_repair(self, result: ValidationResult):
        """Determine appropriate repair action based on errors"""
        hard_errors = result.get_hard_error_types()

        if not hard_errors:
            return

        # Prioritize repairs based on error type
        if "forbidden" in hard_errors:
            result.set_repair(
                RepairType.TEMPLATE_REINFORCE,
                "Remove forbidden content and re-translate"
            )
        elif "format" in hard_errors:
            result.set_repair(
                RepairType.TEMPLATE_REINFORCE,
                "Fix formatting errors and re-translate"
            )
        elif "completion" in hard_errors:
            result.set_repair(
                RepairType.RE_TRANSLATE,
                "Re-translate the chunk completely"
            )
        else:
            result.set_repair(
                RepairType.RE_TRANSLATE,
                "Re-translate with corrections"
            )

    def should_use_llm(self) -> bool:
        """Check if LLM validation should be used"""
        return self.enable_llm_validation

    def set_llm_validation(self, enabled: bool):
        """Enable/disable LLM validation"""
        self.enable_llm_validation = enabled
