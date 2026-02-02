"""
RLM Root Orchestrator
Main orchestration loop with 6-step execution process
"""
from typing import Dict, Any, Optional, List
import time

from rlm_state import TranslationState, PresetType, QualityFlagType, RepairType
from repl_environment_v2 import EnhancedREPL
from sub_translator import SubTranslator
from verifier import Verifier
from context_package import build_context_package


class RootOrchestrator:
    """
    Root Orchestrator - the "supervisor" agent that manages the translation process.
    Executes the Plan→Retrieve→Translate→Verify→Repair→Commit loop.
    """

    def __init__(
        self,
        llm_config,
        preset_type: PresetType = PresetType.GENERAL,
        enable_llm_validation: bool = False,
        max_retries: int = 2,
        source_lang: str = "auto",
        target_lang: str = "ko",
        check_sentence: bool = True,
        check_length: bool = True
    ):
        """
        Initialize root orchestrator.

        Args:
            llm_config: LLM provider configuration
            preset_type: Translation preset type
            enable_llm_validation: Whether to enable LLM validation
            max_retries: Maximum number of retries per chunk
            source_lang: Source language code
            target_lang: Target language code
            check_sentence: Whether to check sentence completion
            check_length: Whether to check translation length
        """
        self.llm_config = llm_config
        self.preset_type = preset_type
        self.max_retries = max_retries
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.check_sentence = check_sentence
        self.check_length = check_length

        # Initialize components
        self.sub_translator = SubTranslator(llm_config, preset_type, source_lang, target_lang)
        self.verifier = Verifier(enable_llm_validation=enable_llm_validation)
        self.repl = None  # Will be initialized when setting text

        # Callback tracking
        self.last_progress_callback = None
        self.last_quality_flags = []
        self.last_cost_stats = (0.0, 0, 0)
        self.last_repair_history = []

    def set_text(self, chunks: List[str]):
        """
        Set the text to translate.

        Args:
            chunks: List of text chunks
        """
        self.repl = EnhancedREPL(
            llm_query_func=self._call_sub_translator,
            preset_type=self.preset_type
        )
        self.repl.set_original_text(chunks)
        self.repl.state.total_chunks = len(chunks)

    def set_glossary(self, glossary: dict):
        """
        Set custom glossary for translation.

        Args:
            glossary: Dictionary mapping source terms to target translations
        """
        if self.repl and glossary:
            for source, target in glossary.items():
                self.repl.state.add_hard_term(source, target)

    def _call_sub_translator(self, chunk: str) -> str:
        """
        Call sub-translator (from REPL context).

        Args:
            chunk: Text to translate

        Returns:
            Translated text
        """
        result = self.sub_translator.translate_chunk(
            chunk_text=chunk,
            chunk_index=self.repl.state.current_chunk_index,
            state=self.repl.state
        )

        if result["success"]:
            return result["translation"]
        else:
            return f"[Translation Error: {result.get('error', 'Unknown error')}]"

    def execute_round(self) -> Dict[str, Any]:
        """
        Execute one round of the 6-step RLM process.

        Plan → Retrieve → Translate → Verify → Repair → Commit

        Returns:
            Dict with round results and status
        """
        start_time = time.time()

        # Step 1: Plan - Select next chunk
        print(f"  [Step 1: PLAN] Selecting next chunk...")
        next_chunk_idx = self.repl._select_next_chunk(strategy="sequential")
        if next_chunk_idx is None:
            print(f"  [Step 1: PLAN] All chunks completed!")
            return {
                "success": True,
                "completed": True,
                "message": "All chunks completed",
                "duration": time.time() - start_time,
            }
        print(f"  [Step 1: PLAN] Selected chunk {next_chunk_idx}")

        # Step 2: Retrieve - Get chunk from REPL
        print(f"  [Step 2: RETRIEVE] Getting chunk {next_chunk_idx} from REPL...")
        chunk = self.repl._get_chunk(next_chunk_idx)
        if chunk is None:
            print(f"  [Step 2: RETRIEVE] ERROR: Chunk not found!")
            return {
                "success": False,
                "completed": False,
                "message": f"Chunk {next_chunk_idx} not found",
                "duration": time.time() - start_time,
            }
        print(f"  [Step 2: RETRIEVE] Got chunk ({len(chunk)} chars)")

        chunk_index = next_chunk_idx

        # Step 3: Translate - Translate chunk with context
        print(f"  [Step 3: TRANSLATE] SubTranslator processing...")
        translation_result = self.sub_translator.translate_chunk(
            chunk_text=chunk,
            chunk_index=chunk_index,
            state=self.repl.state
        )

        if not translation_result["success"]:
            print(f"  [Step 3: TRANSLATE] ERROR: {translation_result.get('error', 'Unknown')}")
            return {
                "success": False,
                "completed": False,
                "message": f"Translation failed for chunk {chunk_index}",
                "error": translation_result.get("error", "Unknown error"),
                "duration": time.time() - start_time,
            }

        translation = translation_result["translation"]
        duration = translation_result.get("duration", 0)
        print(f"  [Step 3: TRANSLATE] Done ({len(translation)} chars, {duration:.2f}s)")

        # Update REPL state
        self.repl.state.add_chunk(chunk, translation)

        # Step 4: Verify - Validate translation
        print(f"  [Step 4: VERIFY] Verifier checking translation...")
        validation_result = self.verifier.validate(
            translation=translation,
            original_chunk=chunk,
            context=build_context_package(
                state=self.repl.state,
                current_chunk_text=chunk,
                current_chunk_index=chunk_index
            ),
            preset_type=self.preset_type.value,
            check_sentence=self.check_sentence,
            check_length=self.check_length
        )
        print(f"  [Step 4: VERIFY] Valid={validation_result.valid}, Issues={len(validation_result.issues) if hasattr(validation_result, 'issues') else 0}")

        # Step 5: Repair - Handle validation failures
        if not validation_result.valid:
            print(f"  [Step 5: REPAIR] Repairing translation...")
            repaired_translation = self._perform_repair(
                chunk_index=chunk_index,
                original_chunk=chunk,
                translation=translation,
                validation_result=validation_result,
                retry_count=0
            )
            print(f"  [Step 5: REPAIR] Repair complete")
        else:
            print(f"  [Step 5: REPAIR] No repair needed (FRESH)")
            repaired_translation = translation

        # Step 6: Commit - Save final translation
        print(f"  [Step 6: COMMIT] Saving translation to REPL state...")
        self.repl._save_translation(
            chunk_index=chunk_index,
            translation=repaired_translation
        )
        print(f"  [Step 6: COMMIT] Done")

        # Add context summary
        context_summary = f"Chunk {chunk_index+1}/{self.repl.state.total_chunks} completed successfully"
        self.repl.state.add_history_summary(context_summary)

        return {
            "success": validation_result.valid,
            "completed": False,  # Not all chunks done yet
            "chunk_index": chunk_index,
            "translation": repaired_translation,
            "validation_passed": validation_result.valid,
            "duration": time.time() - start_time,
            "chunk_duration": duration,
        }

    def _perform_repair(
        self,
        chunk_index: int,
        original_chunk: str,
        translation: str,
        validation_result,
        retry_count: int
    ) -> str:
        """
        Perform repair based on validation results.

        Args:
            chunk_index: Index of chunk
            original_chunk: Original text
            translation: Current translation
            validation_result: Validation result
            retry_count: Current retry count

        Returns:
            Repaired translation
        """
        if retry_count >= self.max_retries:
            print(f"    [REPAIR] Max retries ({self.max_retries}) reached, keeping original translation")
            return translation  # Max retries reached, keep original

        repair_type = validation_result.repair_type
        
        # Log validation errors
        print(f"    [REPAIR] Retry {retry_count + 1}/{self.max_retries}")
        print(f"    [REPAIR] Repair type: {repair_type}")
        if hasattr(validation_result, 'errors') and validation_result.errors:
            for err in validation_result.errors:
                print(f"    [REPAIR] Error: {err.get('message', 'Unknown')}")
        if hasattr(validation_result, 'warnings') and validation_result.warnings:
            for warn in validation_result.warnings:
                print(f"    [REPAIR] Warning: {warn.get('message', 'Unknown')}")

        # Apply repair logic
        if repair_type == RepairType.TEMPLATE_REINFORCE:
            # Re-translate with stronger formatting rules
            print(f"    [REPAIR] Action: Reinforcing formatting...")
            result = self._reinforce_formatting(translation, validation_result)
            print(f"    [REPAIR] Reinforcement complete ({len(result)} chars)")
            return result

        elif repair_type == RepairType.GLOSSARY_UPDATE:
            # Update glossary and re-translate
            print(f"    [REPAIR] Action: Updating glossary and re-translating...")
            result = self._update_glossary_and_retranslate(
                translation, validation_result
            )
            print(f"    [REPAIR] Glossary update complete ({len(result)} chars)")
            return result

        elif repair_type == RepairType.RE_TRANSLATE:
            # Complete re-translation
            print(f"    [REPAIR] Action: Complete re-translation of chunk {chunk_index}...")
            result = self._retranslate_chunk(chunk_index, original_chunk)
            print(f"    [REPAIR] Re-translation complete ({len(result)} chars)")
            return result

        elif repair_type == RepairType.SPLIT_CHUNK:
            # Split chunk and re-translate portion
            print(f"    [REPAIR] Action: Splitting chunk and re-translating...")
            result = self._split_and_retranslate(chunk_index, original_chunk, translation)
            print(f"    [REPAIR] Split re-translation complete ({len(result)} chars)")
            return result

        else:
            # Default: re-translate with context
            print(f"    [REPAIR] Action: Default re-translation...")
            result = self._retranslate_chunk(chunk_index, original_chunk)
            print(f"    [REPAIR] Default re-translation complete ({len(result)} chars)")
            return result

    def _reinforce_formatting(self, translation: str, validation_result) -> str:
        """
        Reinforce formatting by asking sub-translator to fix errors.

        Args:
            translation: Current translation
            validation_result: Validation result

        Returns:
            Re-inforced translation
        """
        # Call sub-translator again with stricter instructions
        result = self.sub_translator.translate_chunk(
            chunk_text=translation,
            chunk_index=self.repl.state.current_chunk_index,
            state=self.repl.state
        )

        if result["success"]:
            self.repl.state.increment_retry_count(QualityFlagType.FORMAT_ERROR)
            return result["translation"]
        else:
            return translation

    def _update_glossary_and_retranslate(
        self,
        translation: str,
        validation_result
    ) -> str:
        """
        Update glossary based on validation and re-translate.

        Args:
            translation: Current translation
            validation_result: Validation result

        Returns:
            Re-translated translation
        """
        # In real implementation, would update glossary with issue details
        # For now, just re-translate
        return self._retranslate_chunk(
            self.repl.state.current_chunk_index,
            self.repl.state.chunk_history[self.repl.state.current_chunk_index]
        )

    def _retranslate_chunk(self, chunk_index: int, original_chunk: str) -> str:
        """
        Re-translate a chunk from scratch.

        Args:
            chunk_index: Index of chunk
            original_chunk: Original text

        Returns:
            New translation
        """
        result = self.sub_translator.translate_chunk(
            chunk_text=original_chunk,
            chunk_index=chunk_index,
            state=self.repl.state
        )

        if result["success"]:
            self.repl.state.increment_retry_count(QualityFlagType.FORMAT_ERROR)
            return result["translation"]
        else:
            return result.get("translation", "")

    def _split_and_retranslate(
        self,
        chunk_index: int,
        original_chunk: str,
        translation: str
    ) -> str:
        """
        Split chunk and re-translate.

        Args:
            chunk_index: Index of chunk
            original_chunk: Original text
            translation: Current translation

        Returns:
            Re-translated translation
        """
        # In real implementation, would split based on issues
        # For now, re-translate completely
        return self._retranslate_chunk(chunk_index, original_chunk)

    def run_full_translation(self, progress_callback=None) -> Dict[str, Any]:
        """
        Run complete translation of all chunks.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with final results and statistics
        """
        if not self.repl:
            return {
                "success": False,
                "message": "No text set for translation",
            }

        total_chunks = self.repl.state.total_chunks
        print(f"[DEBUG] Starting translation of {total_chunks} chunks")
        print(f"[DEBUG] _chunks count: {len(self.repl._chunks)}")
        
        results = []
        error_count = 0
        success_count = 0
        total_duration = 0

        # Keep executing rounds until all chunks are done
        iteration = 0
        max_iterations = total_chunks + 10  # Safety limit
        
        while iteration < max_iterations:
            iteration += 1
            print(f"[DEBUG] Iteration {iteration}, current_chunk_index: {self.repl.state.current_chunk_index}")
            
            # Execute one round
            round_result = self.execute_round()
            
            print(f"[DEBUG] Round result: success={round_result.get('success')}, completed={round_result.get('completed')}, chunk_index={round_result.get('chunk_index')}")
            
            # Check if all chunks are completed
            if round_result.get("completed"):
                print(f"[DEBUG] All chunks completed signal received")
                break
                
            results.append(round_result)
            total_duration += round_result.get("duration", 0)

            if round_result.get("success"):
                success_count += 1
            else:
                error_count += 1
                error_type = round_result.get("message", "Unknown error")
                chunk_idx = round_result.get("chunk_index", success_count + error_count - 1)
                self.repl.state.record_error(
                    chunk_idx, QualityFlagType.FORMAT_ERROR, error_type
                )

            if progress_callback:
                completed = success_count + error_count
                progress = completed / total_chunks if total_chunks > 0 else 1.0
                progress_callback(
                    f"Translating chunk {completed}/{total_chunks}",
                    progress
                )
            
            # Safety check to prevent infinite loop
            if success_count + error_count >= total_chunks:
                print(f"[DEBUG] Reached total_chunks limit: {success_count + error_count} >= {total_chunks}")
                break

        print(f"[DEBUG] Translation loop finished. Success: {success_count}, Errors: {error_count}")

        # Get cost stats from sub_translator
        cost_stats = self.sub_translator.llm_client.cost_summary()
        total_calls = cost_stats.get('total_calls', success_count)  # fallback to success_count
        total_cost = cost_stats.get('total_cost', 0)
        
        print(f"[DEBUG] Cost stats: calls={total_calls}, cost={total_cost}")

        # Final statistics
        final_result = {
            "success": error_count == 0,
            "completed": True,
            "total_chunks": total_chunks,
            "success_chunks": success_count,
            "error_chunks": error_count,
            "total_duration": total_duration,
            "avg_chunk_duration": total_duration / total_chunks if total_chunks > 0 else 0,
            "total_calls": total_calls,
            "total_cost": total_cost,
            "errors": [r for r in results if not r.get("success")],
        }

        return final_result

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress"""
        if not self.repl:
            return {
                "total_chunks": 0,
                "completed_chunks": 0,
                "current_chunk": 0,
                "progress": 0.0,
            }

        total = self.repl.state.total_chunks
        completed = self.repl.state.completed_chunks

        return {
            "total_chunks": total,
            "completed_chunks": completed,
            "current_chunk": self.repl.state.current_chunk_index,
            "progress": completed / total if total > 0 else 0.0,
        }

    def get_final_result(self) -> str:
        """Get final translated text"""
        if not self.repl:
            return ""
        return self.repl.get_translated_text()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get translation state summary"""
        if not self.repl:
            return {}

        return self.repl.get_state_summary()

    def test_connection(self) -> bool:
        """Test LLM connection"""
        return self.sub_translator.test_connection()

    def on_rlm_progress(self, step_name: str, progress: float):
        """
        Callback for RLM step progress updates.

        Args:
            step_name: Name of current step
            progress: Progress value (0.0 to 1.0)
        """
        self.last_progress_callback = (step_name, progress)

    def on_rlm_quality_flags(self, flags: list):
        """
        Callback for quality flags updates.

        Args:
            flags: List of quality flag strings (FRESH, REPAIRED, FAILED)
        """
        self.last_quality_flags = flags

    def on_rlm_cost_stats(self, cost: float, calls: int, chunks: int):
        """
        Callback for cost statistics updates.

        Args:
            cost: Total cost in dollars
            calls: Total number of API calls
            chunks: Number of chunks translated
        """
        self.last_cost_stats = (cost, calls, chunks)

    def on_rlm_repair(self, repair_type: str, message: str):
        """
        Callback for repair history updates.

        Args:
            repair_type: Type of repair performed
            message: Repair message
        """
        if not hasattr(self, 'last_repair_history'):
            self.last_repair_history = []
        self.last_repair_history.append((repair_type, message))