"""
RLM-Trans GUI v2
PyQt6 GUI with preset support and LLM parameter editing
"""
import sys
import json
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QTextEdit, QPushButton, QFileDialog,
    QProgressBar, QGroupBox, QFormLayout, QLineEdit, QTabWidget,
    QMessageBox, QSplitter, QStatusBar, QDialog, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QAction, QPixmap

from config import LLMConfig
from rlm_translator_v2 import RLMTranslatorV2, TranslationResult
from root_orchestrator import RootOrchestrator
from presets_v1 import TranslationPreset, get_preset_manager, LLMParameters
from rlm_state import PresetType
from chunking_strategy import ChunkingStrategy


class PresetEditorDialog(QDialog):
    """Dialog for editing preset settings"""
    
    def __init__(self, preset: TranslationPreset, parent=None):
        super().__init__(parent)
        self.preset = preset
        self.setWindowTitle(f"프리셋 편집: {preset.name}")
        self.setMinimumSize(500, 600)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Scroll area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Basic Info
        info_group = QGroupBox("기본 정보")
        info_layout = QFormLayout()
        
        self.name_edit = QLineEdit(self.preset.name)
        info_layout.addRow("이름:", self.name_edit)
        
        self.desc_edit = QLineEdit(self.preset.description)
        info_layout.addRow("설명:", self.desc_edit)
        
        self.type_edit = QLineEdit(self.preset.document_type)
        info_layout.addRow("문서 유형:", self.type_edit)
        
        info_group.setLayout(info_layout)
        scroll_layout.addWidget(info_group)
        
        # LLM Parameters
        llm_group = QGroupBox("LLM 파라미터")
        llm_layout = QFormLayout()
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(self.preset.llm_params.temperature)
        llm_layout.addRow("Temperature:", self.temp_spin)
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(256, 32768)
        self.max_tokens_spin.setSingleStep(256)
        self.max_tokens_spin.setValue(self.preset.llm_params.max_tokens)
        llm_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setDecimals(2)
        self.top_p_spin.setValue(self.preset.llm_params.top_p)
        llm_layout.addRow("Top P:", self.top_p_spin)
        
        llm_group.setLayout(llm_layout)
        scroll_layout.addWidget(llm_group)
        
        # Translation Settings
        trans_group = QGroupBox("번역 설정")
        trans_layout = QFormLayout()
        
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(500, 10000)
        self.chunk_spin.setSingleStep(100)
        self.chunk_spin.setValue(self.preset.chunk_size)
        trans_layout.addRow("청크 크기:", self.chunk_spin)
        
        self.preserve_format_check = QCheckBox()
        self.preserve_format_check.setChecked(self.preset.preserve_formatting)
        trans_layout.addRow("형식 유지:", self.preserve_format_check)
        
        self.use_glossary_check = QCheckBox()
        self.use_glossary_check.setChecked(self.preset.use_glossary)
        trans_layout.addRow("용어집 사용:", self.use_glossary_check)
        
        trans_group.setLayout(trans_layout)
        scroll_layout.addWidget(trans_group)
        
        # Style Guide
        style_group = QGroupBox("스타일 가이드")
        style_layout = QVBoxLayout()
        
        self.style_edit = QLineEdit(self.preset.style_guide)
        style_layout.addWidget(self.style_edit)
        
        style_group.setLayout(style_layout)
        scroll_layout.addWidget(style_group)
        
        # System Prompt
        prompt_group = QGroupBox("시스템 프롬프트")
        prompt_layout = QVBoxLayout()
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(self.preset.system_prompt)
        self.prompt_edit.setMinimumHeight(150)
        prompt_layout.addWidget(self.prompt_edit)
        
        prompt_group.setLayout(prompt_layout)
        scroll_layout.addWidget(prompt_group)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_updated_preset(self) -> TranslationPreset:
        """Get preset with updated values"""
        self.preset.name = self.name_edit.text()
        self.preset.description = self.desc_edit.text()
        self.preset.document_type = self.type_edit.text()
        
        self.preset.llm_params.temperature = self.temp_spin.value()
        self.preset.llm_params.max_tokens = self.max_tokens_spin.value()
        self.preset.llm_params.top_p = self.top_p_spin.value()
        
        self.preset.chunk_size = self.chunk_spin.value()
        self.preset.preserve_formatting = self.preserve_format_check.isChecked()
        self.preset.use_glossary = self.use_glossary_check.isChecked()
        
        self.preset.style_guide = self.style_edit.text()
        self.preset.system_prompt = self.prompt_edit.toPlainText()
        
        return self.preset


class GlossaryEditorDialog(QDialog):
    """Dialog for editing glossary entries"""
    
    def __init__(self, glossary: dict = None, parent=None):
        super().__init__(parent)
        self.glossary = glossary or {}
        self.setWindowTitle("용어집 편집")
        self.setMinimumSize(600, 500)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel("원본 용어와 번역 용어를 매핑합니다. 번역 시 이 용어집이 우선 적용됩니다.")
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        layout.addWidget(info_label)
        
        # Table for glossary entries
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["원본 (Source)", "번역 (Target)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        
        # Load existing glossary
        self._load_glossary_to_table()
        
        layout.addWidget(self.table)
        
        # Buttons for row management
        row_buttons = QHBoxLayout()
        
        add_row_btn = QPushButton("➕ 행 추가")
        add_row_btn.clicked.connect(self.add_row)
        row_buttons.addWidget(add_row_btn)
        
        remove_row_btn = QPushButton("➖ 선택 삭제")
        remove_row_btn.clicked.connect(self.remove_selected_rows)
        row_buttons.addWidget(remove_row_btn)
        
        clear_btn = QPushButton("🗑️ 전체 삭제")
        clear_btn.clicked.connect(self.clear_all)
        row_buttons.addWidget(clear_btn)
        
        row_buttons.addStretch()
        layout.addLayout(row_buttons)
        
        # Import/Export buttons
        file_buttons = QHBoxLayout()
        
        import_btn = QPushButton("📂 JSON 불러오기")
        import_btn.clicked.connect(self.import_json)
        file_buttons.addWidget(import_btn)
        
        export_btn = QPushButton("💾 JSON 저장")
        export_btn.clicked.connect(self.export_json)
        file_buttons.addWidget(export_btn)
        
        file_buttons.addStretch()
        layout.addLayout(file_buttons)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _load_glossary_to_table(self):
        """Load glossary dictionary to table"""
        from PyQt6.QtWidgets import QTableWidgetItem
        
        self.table.setRowCount(len(self.glossary))
        for i, (source, target) in enumerate(self.glossary.items()):
            self.table.setItem(i, 0, QTableWidgetItem(source))
            self.table.setItem(i, 1, QTableWidgetItem(target))
    
    def add_row(self):
        """Add a new empty row"""
        row = self.table.rowCount()
        self.table.insertRow(row)
    
    def remove_selected_rows(self):
        """Remove selected rows"""
        rows = set(item.row() for item in self.table.selectedItems())
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)
    
    def clear_all(self):
        """Clear all rows"""
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "확인", "모든 용어를 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table.setRowCount(0)
    
    def import_json(self):
        """Import glossary from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "용어집 불러오기", "",
            "JSON 파일 (*.json);;모든 파일 (*.*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Support both flat dict and nested format
                if isinstance(data, dict):
                    if "glossary" in data:
                        glossary = data["glossary"]
                    else:
                        glossary = data
                    
                    # Merge with existing
                    current_count = self.table.rowCount()
                    for source, target in glossary.items():
                        self.table.insertRow(current_count)
                        from PyQt6.QtWidgets import QTableWidgetItem
                        self.table.setItem(current_count, 0, QTableWidgetItem(str(source)))
                        self.table.setItem(current_count, 1, QTableWidgetItem(str(target)))
                        current_count += 1
                    
                    QMessageBox.information(self, "성공", f"{len(glossary)}개 용어를 불러왔습니다.")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"불러오기 실패: {e}")
    
    def export_json(self):
        """Export glossary to JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "용어집 저장", "glossary.json",
            "JSON 파일 (*.json)"
        )
        if file_path:
            try:
                glossary = self.get_glossary()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({"glossary": glossary}, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "성공", f"저장됨: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"저장 실패: {e}")
    
    def get_glossary(self) -> dict:
        """Get glossary dictionary from table"""
        glossary = {}
        for row in range(self.table.rowCount()):
            source_item = self.table.item(row, 0)
            target_item = self.table.item(row, 1)
            
            if source_item and target_item:
                source = source_item.text().strip()
                target = target_item.text().strip()
                if source and target:
                    glossary[source] = target
        
        return glossary

class GlossaryViewerDialog(QDialog):
    """Dialog to view current RLM glossary state (Read-only view of learned terms)"""
    
    def __init__(self, hard_glossary: dict, soft_glossary: dict, confirmed_terms: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RLM 용어집 뷰어 (자동 학습 결과)")
        self.setMinimumSize(700, 600)
        
        self.hard_glossary = hard_glossary
        self.soft_glossary = soft_glossary
        self.confirmed_terms = confirmed_terms
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # Helper to create tabs
        def add_tab(data, title, desc):
            widget = QWidget()
            t_layout = QVBoxLayout(widget)
            
            t_layout.addWidget(QLabel(desc))
            
            from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["원문 (Source)", "번역 (Target)"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            if data:
                table.setRowCount(len(data))
                for i, (k, v) in enumerate(sorted(data.items())):
                    table.setItem(i, 0, QTableWidgetItem(str(k)))
                    table.setItem(i, 1, QTableWidgetItem(str(v)))
            else:
                table.setRowCount(0)
                t_layout.addWidget(QLabel("(데이터 없음)"))
                
            t_layout.addWidget(table)
            tabs.addTab(widget, title)

        add_tab(self.confirmed_terms, "확정 용어 (Confirmed)", "이번 세션에서 확정/학습된 용어 목록입니다.")
        add_tab(self.hard_glossary, "필수 용어 (Hard)", "반드시 지켜야 하는 고정 용어집입니다.")
        add_tab(self.soft_glossary, "참고 용어 (Soft)", "참고용으로 제공된 용어집입니다.")
        
        layout.addWidget(tabs)
        
        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        
        export_btn = QPushButton("JSON으로 내보내기")
        export_btn.clicked.connect(self.export_glossary)
        btn_box.addWidget(export_btn)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        btn_box.addWidget(close_btn)
        
        layout.addLayout(btn_box)
        
    def export_glossary(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "용어집 내보내기", "learned_glossary.json", "JSON Files (*.json)"
        )
        if file_path:
            try:
                data = {
                    "confirmed": self.confirmed_terms,
                    "hard": self.hard_glossary,
                    "soft": self.soft_glossary
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "성공", "용어집이 저장되었습니다.")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"저장 실패: {e}")


class RLMControlPanel(QWidget):
    """Control panel for RLM mode settings with tabbed interface"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # RLM Mode Toggle - prominent
        self.rlm_mode_toggle = QCheckBox("Enable RLM Mode")
        self.rlm_mode_toggle.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                background-color: #e8f5e9;
                border: 2px solid #4CAF50;
                border-radius: 4px;
            }
            QCheckBox:checked {
                background-color: #4CAF50;
                color: white;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        layout.addWidget(self.rlm_mode_toggle)

        # Connect signal
        self.rlm_mode_toggle.toggled.connect(self.on_toggle_changed)

        # Settings Group - compact
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(5, 10, 5, 5)
        settings_layout.setSpacing(5)

        # Max Retries
        retries_row = QHBoxLayout()
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(0, 5)
        self.retries_spin.setValue(2)
        self.retries_spin.setMaximumWidth(60)
        retries_row.addWidget(self.retries_spin)
        self.retries_label = QLabel("times")
        self.retries_label.setStyleSheet("font-size: 10px; color: #666;")
        retries_row.addWidget(self.retries_label)
        retries_row.addStretch()
        
        retries_widget = QWidget()
        retries_widget.setLayout(retries_row)
        settings_layout.addRow("Max Retries:", retries_widget)

        # LLM Validation
        self.llm_validation_check = QCheckBox("LLM Verify")
        self.llm_validation_check.setChecked(True)
        self.llm_validation_check.setStyleSheet("font-size: 10px;")
        settings_layout.addRow("Validation:", self.llm_validation_check)

        # Conflict Resolution
        self.conflict_combo = QComboBox()
        self.conflict_combo.addItems(["PRESET", "DOC_INIT", "MAJORITY", "RECENT"])
        self.conflict_combo.setStyleSheet("font-size: 10px;")
        self.conflict_combo.setMaximumWidth(100)
        settings_layout.addRow("Glossary:", self.conflict_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Chunking Options Group
        chunking_group = QGroupBox("Chunking")
        chunking_layout = QVBoxLayout()
        chunking_layout.setContentsMargins(5, 10, 5, 5)
        chunking_layout.setSpacing(3)
        
        self.paragraph_chunking = QCheckBox("문단 단위 청킹")
        self.paragraph_chunking.setChecked(True)
        self.paragraph_chunking.setToolTip("문단 단위로 청크를 나눕니다. 체크 해제시 단어 수 기반.")
        self.paragraph_chunking.setStyleSheet("font-size: 10px;")
        chunking_layout.addWidget(self.paragraph_chunking)
        
        self.word_chunking = QCheckBox("단어 수 기반 청킹")
        self.word_chunking.setChecked(False)
        self.word_chunking.setToolTip("지정된 단어 수로 청크를 나눕니다.")
        self.word_chunking.setStyleSheet("font-size: 10px;")
        chunking_layout.addWidget(self.word_chunking)
        
        # Make them mutually exclusive
        self.paragraph_chunking.toggled.connect(lambda checked: self.word_chunking.setChecked(not checked) if checked else None)
        self.word_chunking.toggled.connect(lambda checked: self.paragraph_chunking.setChecked(not checked) if checked else None)
        
        chunking_group.setLayout(chunking_layout)
        layout.addWidget(chunking_group)
        
        # Verification Options Group
        verify_group = QGroupBox("누락 확인")
        verify_layout = QVBoxLayout()
        verify_layout.setContentsMargins(5, 10, 5, 5)
        verify_layout.setSpacing(3)
        
        self.sentence_verify = QCheckBox("문장 단위 검토")
        self.sentence_verify.setChecked(True)
        self.sentence_verify.setToolTip("번역이 완전한 문장으로 끝나는지 확인합니다.")
        self.sentence_verify.setStyleSheet("font-size: 10px;")
        verify_layout.addWidget(self.sentence_verify)
        
        self.length_verify = QCheckBox("길이 검토 (50%)")
        self.length_verify.setChecked(False)
        self.length_verify.setToolTip("번역 길이가 원본의 50% 이상인지 확인합니다.")
        self.length_verify.setStyleSheet("font-size: 10px;")
        verify_layout.addWidget(self.length_verify)
        
        verify_group.setLayout(verify_layout)
        layout.addWidget(verify_group)

        layout.addStretch()
        self.setLayout(layout)

    def update_rlm_toggle_style(self):
        """Update RLM toggle style based on state"""
        if self.rlm_mode_toggle.isChecked():
            self.rlm_mode_toggle.setStyleSheet("""
                QCheckBox {
                    color: #2E7D32;
                    font-weight: bold;
                    border: 2px solid #4CAF50;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
            """)
        else:
            self.rlm_mode_toggle.setStyleSheet("""
                QCheckBox {
                    color: #666;
                    border: 2px solid #ccc;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
            """)

    def update_retries_label(self, value: int):
        self.retries_label.setText(f"{value}회")

    def enable_controls(self, enabled: bool):
        self.retries_spin.setEnabled(enabled)
        self.llm_validation_check.setEnabled(enabled)
        self.conflict_combo.setEnabled(enabled)

    def is_rlm_enabled(self) -> bool:
        return self.rlm_mode_toggle.isChecked()

    def get_max_retries(self) -> int:
        return self.retries_spin.value()

    def is_llm_validation_enabled(self) -> bool:
        return self.llm_validation_check.isChecked()

    def get_conflict_resolution(self) -> str:
        return self.conflict_combo.currentText()
    
    def is_paragraph_chunking(self) -> bool:
        return self.paragraph_chunking.isChecked()
    
    def is_sentence_verify(self) -> bool:
        return self.sentence_verify.isChecked()
    
    def is_length_verify(self) -> bool:
        return self.length_verify.isChecked()

    def on_toggle_changed(self, checked: bool):
        """Handle toggle change signal."""
        if self.parent_window:
            self.parent_window.on_rlm_mode_toggled(checked)


class RLMProgressPanel(QWidget):
    """Panel for displaying RLM progress and statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # Use horizontal layout to save vertical space
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Current Step Display
        step_group = QGroupBox("Step")
        step_layout = QVBoxLayout()
        step_layout.setContentsMargins(5, 5, 5, 5)

        self.step_label = QLabel("Ready")
        self.step_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #2196F3;")
        step_layout.addWidget(self.step_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(15)
        step_layout.addWidget(self.progress_bar)

        step_group.setLayout(step_layout)
        main_layout.addWidget(step_group)

        # Quality Flags
        flags_group = QGroupBox("Quality")
        flags_layout = QHBoxLayout()
        flags_layout.setContentsMargins(5, 5, 5, 5)

        self.fresh_flag = QLabel("FRESH")
        self.fresh_flag.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px; padding: 2px 4px; background-color: #E8F5E9; border-radius: 2px;")
        self.fresh_flag.setVisible(False)
        flags_layout.addWidget(self.fresh_flag)

        self.repaired_flag = QLabel("REPAIRED")
        self.repaired_flag.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 10px; padding: 2px 4px; background-color: #FFF3E0; border-radius: 2px;")
        self.repaired_flag.setVisible(False)
        flags_layout.addWidget(self.repaired_flag)

        self.failed_flag = QLabel("FAILED")
        self.failed_flag.setStyleSheet("color: #F44336; font-weight: bold; font-size: 10px; padding: 2px 4px; background-color: #FFEBEE; border-radius: 2px;")
        self.failed_flag.setVisible(False)
        flags_layout.addWidget(self.failed_flag)
        
        flags_layout.addStretch()
        flags_group.setLayout(flags_layout)
        main_layout.addWidget(flags_group)

        # Cost Statistics - compact horizontal
        cost_group = QGroupBox("Cost")
        cost_layout = QHBoxLayout()
        cost_layout.setContentsMargins(5, 5, 5, 5)

        self.total_cost_label = QLabel("$0.00")
        self.total_cost_label.setStyleSheet("font-weight: bold; color: #FF5722; font-size: 10px;")
        cost_layout.addWidget(QLabel("Cost:"))
        cost_layout.addWidget(self.total_cost_label)

        self.total_calls_label = QLabel("0")
        self.total_calls_label.setStyleSheet("color: #607D8B; font-size: 10px;")
        cost_layout.addWidget(QLabel("Calls:"))
        cost_layout.addWidget(self.total_calls_label)

        self.chunks_label = QLabel("0")
        self.chunks_label.setStyleSheet("color: #607D8B; font-size: 10px;")
        cost_layout.addWidget(QLabel("Chunks:"))
        cost_layout.addWidget(self.chunks_label)

        cost_group.setLayout(cost_layout)
        main_layout.addWidget(cost_group)

        # Repair History - compact
        repair_group = QGroupBox("Repair")
        repair_layout = QVBoxLayout()
        repair_layout.setContentsMargins(5, 5, 5, 5)

        self.repair_history_label = QLabel("None")
        self.repair_history_label.setStyleSheet("color: #666; font-size: 10px;")
        self.repair_history_label.setWordWrap(True)
        repair_layout.addWidget(self.repair_history_label)

        repair_group.setLayout(repair_layout)
        main_layout.addWidget(repair_group)

        self.setLayout(main_layout)

    def update_step(self, step_name: str):
        self.step_label.setText(step_name)
        self.progress_bar.setVisible(True)

    def update_progress(self, value: float):
        self.progress_bar.setValue(int(value * 100))

    def update_quality_flags(self, flags: list):
        self.fresh_flag.setVisible(False)
        self.repaired_flag.setVisible(False)
        self.failed_flag.setVisible(False)

        for flag in flags:
            if flag == "FRESH":
                self.fresh_flag.setVisible(True)
            elif flag == "REPAIRED":
                self.repaired_flag.setVisible(True)
            elif flag == "FAILED":
                self.failed_flag.setVisible(True)

    def update_cost_stats(self, cost: float, calls: int, chunks: int):
        self.total_cost_label.setText(f"{cost:.4f}")
        self.total_calls_label.setText(str(calls))
        self.chunks_label.setText(str(chunks))

    def add_repair_history(self, repair_type: str, message: str):
        current = self.repair_history_label.text()
        if current == "None":
            self.repair_history_label.setText(f"[{repair_type}] {message}")
        else:
            self.repair_history_label.setText(
                f"{current}; [{repair_type}] {message}"
            )

    def clear(self):
        self.step_label.setText("Ready")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.fresh_flag.setVisible(False)
        self.repaired_flag.setVisible(False)
        self.failed_flag.setVisible(False)
        self.total_cost_label.setText("$0.00")
        self.total_calls_label.setText("0")
        self.chunks_label.setText("0")
        self.repair_history_label.setText("None")


class TranslationWorker(QThread):
    """Worker thread for translation"""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    rlm_step = pyqtSignal(str)
    rlm_quality_flags = pyqtSignal(list)
    rlm_cost_stats = pyqtSignal(float, int, int)

    def __init__(self, translator, text: str,
                 source_lang: str, target_lang: str, use_rlm: bool = False):
        super().__init__()
        self.translator = translator
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_rlm = use_rlm

    def run(self):
        try:
            if self.use_rlm:
                # RLM mode - use RootOrchestrator
                def progress_callback(msg, prog):
                    self.progress.emit(msg, prog)
                
                result_dict = self.translator.run_full_translation(progress_callback)
                
                # Create a simple result object with all required attributes
                final_text = self.translator.get_final_result()
                total_chunks = result_dict.get('total_chunks', 0)
                total_cost = result_dict.get('total_cost', 0)
                total_calls = result_dict.get('total_calls', 0)
                
                result = type('TranslationResult', (), {
                    'success': True,
                    'translated_text': final_text,
                    'chunks_count': total_chunks,
                    'total_cost': total_cost,
                    'total_time': result_dict.get('total_time', 0),
                    'preset_used': 'RLM Mode',
                    'cost_summary': {
                        'total_calls': total_calls,
                        'total_cost': total_cost
                    },
                    'error_message': None
                })()
                self.finished.emit(result)
            else:
                # Non-RLM mode - use RLMTranslatorV2
                self.translator.progress_callback = lambda msg, prog: self.progress.emit(msg, prog)

                result = self.translator.translate(
                    self.text,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang
                )
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class RLMTranslatorGUIv2(QMainWindow):
    """Main GUI window v2 with preset support"""
    
    def __init__(self):
        super().__init__()
        self.translator: Optional[RLMTranslatorV2] = None
        self.root_orchestrator: Optional[RootOrchestrator] = None
        self.worker: Optional[TranslationWorker] = None
        self.current_file: Optional[Path] = None
        self.preset_manager = get_preset_manager()
        self.use_rlm_mode: bool = False
        self.custom_glossary: dict = {}  # User-defined glossary

        self.init_ui()
        self.init_translator()

        # Hide RLM advanced panels by default (toggle visible via checkbox)
        self.rlm_control_panel.setVisible(True)
        self.rlm_progress_panel.setVisible(False)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("RLM-Trans v2 - 프리셋 지원 번역기")
        self.setMinimumSize(1000, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top controls
        controls_layout = QHBoxLayout()

        # Preset selection
        preset_group = QGroupBox("프리셋")
        preset_layout = QVBoxLayout()

        preset_select_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(200)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed_in_gui)
        preset_select_layout.addWidget(self.preset_combo)

        self.edit_preset_btn = QPushButton("편집")
        self.edit_preset_btn.clicked.connect(self.edit_preset)
        preset_select_layout.addWidget(self.edit_preset_btn)

        self.save_preset_btn = QPushButton("저장")
        self.save_preset_btn.clicked.connect(self.save_preset)
        preset_select_layout.addWidget(self.save_preset_btn)

        preset_layout.addLayout(preset_select_layout)

        # Preset info - more visible
        self.preset_info_label = QLabel("No preset selected")
        self.preset_info_label.setStyleSheet("""
            QLabel {
                color: #1976D2;
                font-size: 11px;
                font-weight: bold;
                padding: 3px 6px;
                background-color: #E3F2FD;
                border-radius: 3px;
            }
        """)
        preset_layout.addWidget(self.preset_info_label)

        preset_group.setLayout(preset_layout)
        controls_layout.addWidget(preset_group)

        # Language selection
        lang_group = QGroupBox("언어")
        lang_layout = QFormLayout()

        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["자동 감지", "한국어", "일본어", "영어"])
        lang_layout.addRow("원본:", self.source_lang_combo)

        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["한국어", "일본어", "영어"])
        lang_layout.addRow("대상:", self.target_lang_combo)

        lang_group.setLayout(lang_layout)
        controls_layout.addWidget(lang_group)

        # LLM Settings
        llm_group = QGroupBox("LLM")
        llm_layout = QFormLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["LM Studio", "OpenAI", "Gemini"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        llm_layout.addRow("프로바이더:", self.provider_combo)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        llm_layout.addRow("모델:", self.model_combo)

        self.test_btn = QPushButton("연결 테스트")
        self.test_btn.clicked.connect(self.test_connection)
        llm_layout.addRow("", self.test_btn)

        llm_group.setLayout(llm_layout)
        controls_layout.addWidget(llm_group)

        # Quick LLM params display
        params_group = QGroupBox("LLM 파라미터")
        params_layout = QFormLayout()

        self.temp_display = QLabel("0.3")
        params_layout.addRow("Temp:", self.temp_display)

        self.tokens_display = QLabel("4096")
        params_layout.addRow("Tokens:", self.tokens_display)

        self.chunk_display = QLabel("2000")
        params_layout.addRow("Chunk:", self.chunk_display)

        params_group.setLayout(params_layout)
        controls_layout.addWidget(params_group)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Main content area with RLM panel on left
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # RLM Control Panel (Left Side)
        self.rlm_control_panel = RLMControlPanel(parent=self)
        self.rlm_control_panel.setMaximumWidth(280)
        content_splitter.addWidget(self.rlm_control_panel)

        # Text areas (Center)
        text_splitter = QSplitter(Qt.Orientation.Vertical)

        # Source text
        source_widget = QWidget()
        source_layout = QVBoxLayout(source_widget)
        source_layout.setContentsMargins(5, 5, 5, 5)

        source_header = QHBoxLayout()
        source_header.addWidget(QLabel("원문"))
        self.load_btn = QPushButton("파일 불러오기")
        self.load_btn.clicked.connect(self.load_file)
        source_header.addWidget(self.load_btn)
        
        self.glossary_btn = QPushButton("📖 용어집")
        self.glossary_btn.clicked.connect(self.edit_glossary)
        self.glossary_btn.setToolTip("용어집 편집 - 일관된 용어 번역을 위해 사용")
        source_header.addWidget(self.glossary_btn)
        
        source_header.addStretch()
        self.char_count_label = QLabel("0자")
        source_header.addWidget(self.char_count_label)
        source_layout.addLayout(source_header)

        self.source_text = QTextEdit()
        self.source_text.setFont(QFont("Malgun Gothic", 11))
        self.source_text.setPlaceholderText("번역할 텍스트를 입력하거나 파일을 불러오세요...")
        self.source_text.textChanged.connect(self.update_char_count)
        source_layout.addWidget(self.source_text)

        text_splitter.addWidget(source_widget)

        # Target text
        target_widget = QWidget()
        target_layout = QVBoxLayout(target_widget)
        target_layout.setContentsMargins(5, 5, 5, 5)

        target_header = QHBoxLayout()
        target_header.addWidget(QLabel("번역 결과"))
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_file)
        self.save_btn.setEnabled(False)
        target_header.addWidget(self.save_btn)

        self.glossary_view_btn = QPushButton("용어집 보기")
        self.glossary_view_btn.clicked.connect(self.view_glossary)
        target_header.addWidget(self.glossary_view_btn)
        self.copy_btn = QPushButton("복사")
        self.copy_btn.clicked.connect(self.copy_result)
        self.copy_btn.setEnabled(False)
        target_header.addWidget(self.copy_btn)
        target_header.addStretch()
        target_layout.addLayout(target_header)

        self.target_text = QTextEdit()
        self.target_text.setFont(QFont("Malgun Gothic", 11))
        self.target_text.setReadOnly(True)
        target_layout.addWidget(self.target_text)

        text_splitter.addWidget(target_widget)
        content_splitter.addWidget(text_splitter)
        main_layout.addWidget(content_splitter, 1)

        # RLM Progress Panel (Bottom) - compact horizontal layout
        self.rlm_progress_panel = RLMProgressPanel()
        self.rlm_progress_panel.setMinimumHeight(60)
        self.rlm_progress_panel.setMaximumHeight(80)
        main_layout.addWidget(self.rlm_progress_panel)

        # Progress bar (original, for non-RLM mode)
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)
        
        # Translate button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.translate_btn = QPushButton("번역 시작")
        self.translate_btn.setMinimumWidth(150)
        self.translate_btn.setMinimumHeight(40)
        self.translate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.translate_btn.clicked.connect(self.start_translation)
        button_layout.addWidget(self.translate_btn)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_translation)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Menu bar
        self.create_menu()
        
        # Load presets into combo
        self.refresh_presets()
    
    def create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("파일")
        
        open_action = QAction("열기...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("저장...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Preset menu
        preset_menu = menubar.addMenu("프리셋")
        
        new_preset_action = QAction("새 프리셋...", self)
        new_preset_action.triggered.connect(self.create_new_preset)
        preset_menu.addAction(new_preset_action)
        
        import_action = QAction("프리셋 가져오기...", self)
        import_action.triggered.connect(self.import_preset)
        preset_menu.addAction(import_action)
        
        export_action = QAction("프리셋 내보내기...", self)
        export_action.triggered.connect(self.export_preset)
        preset_menu.addAction(export_action)
        
        # Help menu
        help_menu = menubar.addMenu("도움말")
        about_action = QAction("정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def refresh_presets(self):
        """Refresh preset combo box"""
        self.preset_combo.clear()
        for preset in self.preset_manager.list_presets_with_info():
            self.preset_combo.addItem(preset["name"], preset["key"])
    
    def init_translator(self):
        """Initialize the translator"""
        try:
            config = LLMConfig.from_env()
            self.translator = RLMTranslatorV2(llm_config=config, preset_name="general")
            
            provider_map = {"lmstudio": 0, "openai": 1, "gemini": 2}
            self.provider_combo.setCurrentIndex(provider_map.get(config.provider, 0))
            
            self.refresh_models()
            self.update_preset_display()
            self.status_bar.showMessage("준비됨")
            
        except Exception as e:
            self.status_bar.showMessage(f"초기화 오류: {e}")
    
    def on_preset_changed(self, preset_name: str):
        """Handle preset selection change"""
        key = self.preset_combo.currentData()
        if key and self.translator:
            self.translator.set_preset(key)
            self.update_preset_display()
    
    def update_preset_display(self):
        """Update preset info display"""
        if not self.translator:
            return
        
        info = self.translator.get_preset_info()
        if info:
            name = info.get('name', 'Unknown')
            desc = info.get('description', '')
            self.preset_info_label.setText(f"📋 {name}")
            self.preset_info_label.setToolTip(desc)
            self.temp_display.setText(f"{info.get('temperature', 0.3):.2f}")
            self.tokens_display.setText(str(info.get('max_tokens', 4096)))
            self.chunk_display.setText(str(info.get('chunk_size', 2000)))
            self.status_bar.showMessage(f"프리셋: {name}")

    def save_file(self):
        """Save translation to file with auto-incrementing filename check"""
        if not self.target_text.toPlainText():
            return
            
        initial_name = "translation_result.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "저장", initial_name, "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            path_obj = Path(file_path)
            
            # Auto-increment if file exists (Update request)
            counter = 1
            while path_obj.exists():
                stem = path_obj.stem
                import re
                match = re.search(r'_(\d+)$', stem)
                
                if match:
                   # This handles "test_1" -> "test_2"
                   base_name = stem[:match.start()]
                   current_num = int(match.group(1))
                   path_obj = path_obj.with_name(f"{base_name}_{current_num + 1}{path_obj.suffix}")
                else:
                   path_obj = path_obj.with_name(f"{stem}_{counter}{path_obj.suffix}")
                
            try:
                with open(path_obj, 'w', encoding='utf-8') as f:
                    f.write(self.target_text.toPlainText())
                QMessageBox.information(self, "저장됨", f"저장되었습니다: {path_obj.name}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"저장 실패: {e}")

    def copy_result(self):
        """Copy translation to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.target_text.toPlainText())
        self.status_bar.showMessage("클립보드에 복사됨", 2000)

    def view_glossary(self):
        """Open glossary viewer dialog"""
        if not self.translator:
             QMessageBox.information(self, "알림", "번역기가 초기화되지 않았습니다.")
             return

        # Try to get state from RLM
        hard = {}
        soft = {}
        confirmed = {}
        
        if hasattr(self.translator, 'root_agent') and hasattr(self.translator.root_agent, 'repl'):
            if self.translator.root_agent.repl and self.translator.root_agent.repl.state:
                state = self.translator.root_agent.repl.state
                hard = state.hard_glossary
                soft = state.soft_glossary
                confirmed = state.confirmed_terms
            
        dialog = GlossaryViewerDialog(hard, soft, confirmed, self)
        dialog.exec()
    
    def edit_preset(self):
        """Open preset editor dialog"""
        if not self.translator or not self.translator.current_preset:
            return
        
        # Create a copy for editing
        preset = TranslationPreset.from_dict(self.translator.current_preset.to_dict())
        
        dialog = PresetEditorDialog(preset, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated = dialog.get_updated_preset()
            key = self.preset_combo.currentData()
            self.preset_manager.save_preset(key, updated)
            self.translator.set_preset(key)
            self.update_preset_display()
            QMessageBox.information(self, "저장됨", "프리셋이 저장되었습니다.")
    
    def edit_glossary(self):
        """Open glossary editor dialog"""
        dialog = GlossaryEditorDialog(self.custom_glossary.copy(), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_glossary = dialog.get_glossary()
            term_count = len(self.custom_glossary)
            if term_count > 0:
                self.glossary_btn.setText(f"📖 용어집 ({term_count})")
                self.status_bar.showMessage(f"용어집: {term_count}개 용어 적용됨")
            else:
                self.glossary_btn.setText("📖 용어집")
                self.status_bar.showMessage("용어집 비어있음")
    
    def save_preset(self):
        """Save current preset as new"""
        if not self.translator:
            return
        
        from PyQt6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "프리셋 저장", "새 프리셋 이름:")
        if ok and name:
            key = name.lower().replace(" ", "_")
            self.translator.save_current_preset_as(key, name)
            self.refresh_presets()
            QMessageBox.information(self, "저장됨", f"'{name}' 프리셋이 저장되었습니다.")
    
    def create_new_preset(self):
        """Create new preset from scratch"""
        from PyQt6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(self, "새 프리셋", "프리셋 이름:")
        if ok and name:
            key = name.lower().replace(" ", "_")
            self.preset_manager.create_custom_preset(key, name, base_preset="general")
            self.refresh_presets()
            
            # Select the new preset
            index = self.preset_combo.findData(key)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
    
    def import_preset(self):
        """Import preset from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "프리셋 가져오기", "", "JSON Files (*.json)"
        )
        if file_path:
            preset = self.preset_manager.import_preset(Path(file_path))
            if preset:
                self.refresh_presets()
                QMessageBox.information(self, "가져오기 완료", f"'{preset.name}' 프리셋을 가져왔습니다.")
    
    def export_preset(self):
        """Export current preset to JSON file"""
        if not self.translator:
            return
        
        key = self.preset_combo.currentData()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "프리셋 내보내기", f"{key}.json", "JSON Files (*.json)"
        )
        if file_path:
            self.preset_manager.export_preset(key, Path(file_path))
            QMessageBox.information(self, "내보내기 완료", "프리셋을 내보냈습니다.")
    
    def on_provider_changed(self, provider_name: str):
        provider_map = {"LM Studio": "lmstudio", "OpenAI": "openai", "Gemini": "gemini"}
        provider = provider_map.get(provider_name, "lmstudio")

        config = LLMConfig.from_env()
        config.provider = provider

        try:
            if self.use_rlm_mode:
                # Use RLM mode with RootOrchestrator
                from rlm_state import PresetType
                preset_key = self.preset_combo.currentData() or "general"
                preset_type = self._get_preset_type(preset_key)
                self.root_orchestrator = RootOrchestrator(
                    llm_config=config,
                    preset_type=preset_type,
                    max_retries=self.rlm_control_panel.get_max_retries()
                )
                self.translator = None
            else:
                # Use non-RLM mode with RLMTranslatorV2
                preset_key = self.preset_combo.currentData() or "general"
                self.translator = RLMTranslatorV2(llm_config=config, preset_name=preset_key)
                self.root_orchestrator = None

            self.refresh_models()
            self.status_bar.showMessage(f"{provider_name}로 변경됨")
        except Exception as e:
            self.status_bar.showMessage(f"프로바이더 변경 실패: {e}")
    
    def refresh_models(self):
        self.model_combo.clear()
        if self.translator:
            try:
                models = self.translator.list_models()
                if models:
                    self.model_combo.addItems(models)
                else:
                    self.model_combo.addItem("(모델 없음)")
            except:
                self.model_combo.addItem("(연결 실패)")
    
    def on_model_changed(self, model_name: str):
        """Handle model selection change - load model in LM Studio if needed"""
        if not model_name or model_name.startswith("("):
            return  # Ignore placeholder items
        
        # Only for LM Studio provider
        if self.provider_combo.currentText() != "LM Studio":
            return
        
        if not self.translator:
            return
        
        try:
            # Check if we have llm_client with ensure_model_loaded
            if hasattr(self.translator, 'llm_client'):
                llm_client = self.translator.llm_client
                if hasattr(llm_client, 'ensure_model_loaded'):
                    self.status_bar.showMessage(f"모델 로드 중: {model_name}...")
                    QApplication.processEvents()  # Update UI
                    
                    if llm_client.ensure_model_loaded(model_name):
                        self.status_bar.showMessage(f"모델 로드됨: {model_name}")
                    else:
                        self.status_bar.showMessage(f"모델 로드 실패: {model_name}")
        except Exception as e:
            self.status_bar.showMessage(f"모델 변경 실패: {e}")
    
    def test_connection(self):
        if not self.translator:
            return
        
        if self.translator.test_connection():
            QMessageBox.information(self, "성공", "LLM 서버에 연결되었습니다.")
            self.refresh_models()
        else:
            QMessageBox.warning(self, "실패", "연결할 수 없습니다.")
    
    def update_char_count(self):
        text = self.source_text.toPlainText()
        self.char_count_label.setText(f"{len(text)}자")
    
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "파일 열기", "",
            "텍스트 파일 (*.txt *.srt *.md);;모든 파일 (*.*)"
        )
        if file_path:
            # Try multiple encodings for Korean text files
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-16', 'latin-1']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    QMessageBox.warning(self, "오류", f"파일 열기 실패: {e}")
                    return
            
            if content is not None:
                self.source_text.setPlainText(content)
                self.current_file = Path(file_path)
                
                # Auto-select subtitle preset for .srt files
                if file_path.endswith('.srt'):
                    index = self.preset_combo.findData("subtitle")
                    if index >= 0:
                        self.preset_combo.setCurrentIndex(index)
                
                self.status_bar.showMessage(f"로드됨: {file_path} ({used_encoding})")
            else:
                QMessageBox.warning(self, "오류", "파일 인코딩을 인식할 수 없습니다.")
    
    def save_file(self):
        if not self.target_text.toPlainText():
            return
        
        suggested = ""
        if self.current_file:
            target_lang = self._get_target_lang_code()
            suggested = str(self.current_file.parent / f"{self.current_file.stem}_{target_lang}{self.current_file.suffix}")
        
        file_path, _ = QFileDialog.getSaveFileName(self, "저장", suggested, "텍스트 파일 (*.txt);;SRT (*.srt)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.target_text.toPlainText())
            self.status_bar.showMessage(f"저장됨: {file_path}")
    
    def copy_result(self):
        QApplication.clipboard().setText(self.target_text.toPlainText())
        self.status_bar.showMessage("복사됨")
    
    def _get_source_lang_code(self) -> str:
        lang_map = {"자동 감지": "auto", "한국어": "ko", "일본어": "ja", "영어": "en"}
        return lang_map.get(self.source_lang_combo.currentText(), "auto")
    
    def _get_target_lang_code(self) -> str:
        lang_map = {"한국어": "ko", "일본어": "ja", "영어": "en"}
        return lang_map.get(self.target_lang_combo.currentText(), "ko")

    def _get_preset_type(self, preset_key: str) -> PresetType:
        """Convert preset key string to PresetType enum."""
        preset_map = {
            "subtitle": PresetType.SUBTITLE,
            "patent": PresetType.PATENT,
            "paper": PresetType.PAPER,
            "novel": PresetType.NOVEL,
            "technical": PresetType.TECHNICAL,
            "general": PresetType.GENERAL
        }
        return preset_map.get(preset_key, PresetType.GENERAL)

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        """Split text into chunks based on selected chunking option."""
        chunker = ChunkingStrategy(chunk_size=chunk_size)
        
        # Check which chunking mode is selected
        if self.rlm_control_panel.is_paragraph_chunking():
            # Paragraph-based chunking
            def show_warning(msg):
                print(f"[CHUNKING WARNING] {msg}")
                # Could also show in status bar
            chunks = chunker.chunk_by_paragraph(text, show_warning_callback=show_warning)
        else:
            # Word/character-based chunking
            chunks = chunker.chunk_text(text)
        
        return [chunk[2] for chunk in chunks]

    def start_translation(self):
        text = self.source_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "오류", "텍스트를 입력해주세요.")
            return

        # Clear previous state
        self.target_text.clear()
        self.rlm_progress_panel.clear()
        self.translate_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Initialize RLM orchestrator if RLM mode is enabled
        if self.use_rlm_mode:
            preset_key = self.preset_combo.currentData() or "general"
            try:
                config = LLMConfig.from_env()
                preset_type = self._get_preset_type(preset_key)
                self.root_orchestrator = RootOrchestrator(
                    llm_config=config,
                    preset_type=preset_type,
                    max_retries=self.rlm_control_panel.get_max_retries(),
                    source_lang=self._get_source_lang_code(),
                    target_lang=self._get_target_lang_code(),
                    check_sentence=self.rlm_control_panel.is_sentence_verify(),
                    check_length=self.rlm_control_panel.is_length_verify()
                )

                # Chunk text
                chunks = self._chunk_text(text)

                # Set text
                self.root_orchestrator.set_text(chunks)
                
                # Set custom glossary if defined
                if self.custom_glossary:
                    self.root_orchestrator.set_glossary(self.custom_glossary)
                    print(f"[GLOSSARY] {len(self.custom_glossary)} terms loaded")

                # Create worker with RLM support
                self.worker = TranslationWorker(
                    self.root_orchestrator, text,
                    self._get_source_lang_code(),
                    self._get_target_lang_code(),
                    use_rlm=True
                )
            except Exception as e:
                QMessageBox.warning(self, "오류", f"RLM 초기화 실패: {e}")
                self.cancel_translation()
                return
        else:
            # Non-RLM mode (existing behavior)
            if not self.translator:
                QMessageBox.warning(self, "오류", "번역기가 초기화되지 않았습니다.")
                return

            self.translator.reset_costs()
            self.worker = TranslationWorker(
                self.translator, text,
                self._get_source_lang_code(),
                self._get_target_lang_code(),
                use_rlm=False
            )

        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def cancel_translation(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.on_finished(None)
    
    def on_progress(self, message: str, progress: float):
        self.progress_bar.setValue(int(progress * 100))
        self.progress_label.setText(message)
    
    def on_finished(self, result: Optional[TranslationResult]):
        self.translate_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        if result is None:
            return
        
        if result.success:
            self.target_text.setPlainText(result.translated_text)
            self.save_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            
            cost = result.cost_summary
            self.status_bar.showMessage(
                f"완료 [{result.preset_used}] - {result.chunks_count}청크, {cost['total_calls']}회 호출"
            )
        else:
            if result.translated_text:
                self.target_text.setPlainText(result.translated_text)
            QMessageBox.warning(self, "오류", f"번역 오류: {result.error_message}")
    
    def on_error(self, error: str):
        self.translate_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "오류", f"번역 실패: {error}")
    
    def show_about(self):
        QMessageBox.about(
            self, "RLM-Trans v2",
            "RLM-Trans v2.0\n\n"
            "재귀 언어 모델 기반 번역기\n"
            "프리셋 시스템 지원\n\n"
            "문서 유형별 최적화 번역:\n"
            "- 자막, 논문, 특허, 소설, 기술 문서\n\n"
            "RLM 기능:\n"
            "- 6단계 번역 루프\n"
            "- 자동 복구\n"
            "- 품질 플래그 추적\n"
            "- 비용 통계"
        )

    def on_rlm_progress(self, step_name: str, progress: float):
        """
        Handle RLM progress updates.

        Args:
            step_name: Name of the current step (PLAN, RETRIEVE, etc.)
            progress: Progress value (0.0 to 1.0)
        """
        self.rlm_progress_panel.update_step(step_name)
        self.rlm_progress_panel.update_progress(progress)
        self.progress_label.setText(step_name)

    def on_rlm_quality_flags(self, flags: list):
        """
        Handle RLM quality flags updates.

        Args:
            flags: List of quality flags (FRESH, REPAIRED, FAILED)
        """
        self.rlm_progress_panel.update_quality_flags(flags)

    def on_rlm_cost_stats(self, cost: float, calls: int, chunks: int):
        """
        Handle RLM cost statistics updates.

        Args:
            cost: Total cost in dollars
            calls: Total number of API calls
            chunks: Number of chunks translated
        """
        self.rlm_progress_panel.update_cost_stats(cost, calls, chunks)

    def on_rlm_repair(self, repair_type: str, message: str):
        """
        Handle RLM repair history updates.

        Args:
            repair_type: Type of repair performed
            message: Repair message
        """
        self.rlm_progress_panel.add_repair_history(repair_type, message)

    def on_preset_changed_in_gui(self, preset_name: str):
        """
        Handle preset selection change (for non-RLM mode).

        Args:
            preset_name: Name of selected preset
        """
        key = self.preset_combo.currentData()
        if key and self.translator:
            self.translator.set_preset(key)
            self.update_preset_display()

    def on_rlm_mode_toggled(self, enabled: bool):
        """
        Handle RLM mode toggle.

        Args:
            enabled: True if RLM mode is enabled
        """
        self.use_rlm_mode = enabled

        if enabled:
            # Show RLM progress panel when enabled
            self.rlm_progress_panel.setVisible(True)
            self.status_bar.showMessage("RLM 모드 활성화됨 - 6단계 번역 루프 사용")
        else:
            # Hide progress panel when disabled
            self.rlm_progress_panel.setVisible(False)
            self.rlm_progress_panel.clear()
            self.status_bar.showMessage("기존 번역기 사용 중")

    def set_rlm_controls(self, enabled: bool):
        """Enable/disable RLM controls."""
        self.rlm_control_panel.enable_controls(enabled)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = RLMTranslatorGUIv2()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
