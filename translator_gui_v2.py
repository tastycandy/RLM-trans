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
from PyQt6.QtGui import QFont, QAction

from config import LLMConfig
from rlm_translator_v2 import RLMTranslatorV2, TranslationResult
from presets_v1 import TranslationPreset, get_preset_manager, LLMParameters


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


class TranslationWorker(QThread):
    """Worker thread for translation"""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, translator: RLMTranslatorV2, text: str, 
                 source_lang: str, target_lang: str):
        super().__init__()
        self.translator = translator
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def run(self):
        try:
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
        self.worker: Optional[TranslationWorker] = None
        self.current_file: Optional[Path] = None
        self.preset_manager = get_preset_manager()
        
        self.init_ui()
        self.init_translator()
        
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
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_select_layout.addWidget(self.preset_combo)
        
        self.edit_preset_btn = QPushButton("편집")
        self.edit_preset_btn.clicked.connect(self.edit_preset)
        preset_select_layout.addWidget(self.edit_preset_btn)
        
        self.save_preset_btn = QPushButton("저장")
        self.save_preset_btn.clicked.connect(self.save_preset)
        preset_select_layout.addWidget(self.save_preset_btn)
        
        preset_layout.addLayout(preset_select_layout)
        
        # Preset info
        self.preset_info_label = QLabel("")
        self.preset_info_label.setStyleSheet("color: #666; font-size: 11px;")
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
        
        # Text areas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Source text
        source_widget = QWidget()
        source_layout = QVBoxLayout(source_widget)
        source_layout.setContentsMargins(0, 0, 0, 0)
        
        source_header = QHBoxLayout()
        source_header.addWidget(QLabel("원문"))
        self.load_btn = QPushButton("파일 불러오기")
        self.load_btn.clicked.connect(self.load_file)
        source_header.addWidget(self.load_btn)
        source_header.addStretch()
        self.char_count_label = QLabel("0자")
        source_header.addWidget(self.char_count_label)
        source_layout.addLayout(source_header)
        
        self.source_text = QTextEdit()
        self.source_text.setFont(QFont("Malgun Gothic", 11))
        self.source_text.setPlaceholderText("번역할 텍스트를 입력하거나 파일을 불러오세요...")
        self.source_text.textChanged.connect(self.update_char_count)
        source_layout.addWidget(self.source_text)
        
        splitter.addWidget(source_widget)
        
        # Target text
        target_widget = QWidget()
        target_layout = QVBoxLayout(target_widget)
        target_layout.setContentsMargins(0, 0, 0, 0)
        
        target_header = QHBoxLayout()
        target_header.addWidget(QLabel("번역 결과"))
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_file)
        self.save_btn.setEnabled(False)
        target_header.addWidget(self.save_btn)
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
        
        splitter.addWidget(target_widget)
        main_layout.addWidget(splitter, 1)
        
        # Progress bar
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
            self.preset_info_label.setText(f"{info.get('description', '')}")
            self.temp_display.setText(f"{info.get('temperature', 0.3):.2f}")
            self.tokens_display.setText(str(info.get('max_tokens', 4096)))
            self.chunk_display.setText(str(info.get('chunk_size', 2000)))
    
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
            preset_key = self.preset_combo.currentData() or "general"
            self.translator = RLMTranslatorV2(llm_config=config, preset_name=preset_key)
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
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.source_text.setPlainText(content)
                self.current_file = Path(file_path)
                
                # Auto-select subtitle preset for .srt files
                if file_path.endswith('.srt'):
                    index = self.preset_combo.findData("subtitle")
                    if index >= 0:
                        self.preset_combo.setCurrentIndex(index)
                
                self.status_bar.showMessage(f"로드됨: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"파일 열기 실패: {e}")
    
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
    
    def start_translation(self):
        text = self.source_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "오류", "텍스트를 입력해주세요.")
            return
        
        if not self.translator:
            QMessageBox.warning(self, "오류", "번역기가 초기화되지 않았습니다.")
            return
        
        self.translator.reset_costs()
        
        self.translate_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.target_text.clear()
        
        self.worker = TranslationWorker(
            self.translator, text,
            self._get_source_lang_code(),
            self._get_target_lang_code()
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
            "- 자막, 논문, 특허, 소설, 기술 문서"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = RLMTranslatorGUIv2()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
