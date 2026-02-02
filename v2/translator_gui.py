"""
RLM-Trans GUI
PyQt6 based graphical user interface for RLM translator
"""
import sys
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QTextEdit, QPushButton, QFileDialog,
    QProgressBar, QGroupBox, QFormLayout, QLineEdit, QTabWidget,
    QMessageBox, QSplitter, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QAction

from config import LLMConfig, TranslationConfig, LANGUAGE_NAMES_KO
from rlm_translator import RLMTranslator, TranslationResult


class TranslationWorker(QThread):
    """Worker thread for translation to keep UI responsive"""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, translator: RLMTranslator, text: str, 
                 source_lang: str, target_lang: str):
        super().__init__()
        self.translator = translator
        self.text = text
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def run(self):
        try:
            # Set progress callback
            self.translator.progress_callback = lambda msg, prog: self.progress.emit(msg, prog)
            
            result = self.translator.translate(
                self.text,
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class RLMTranslatorGUI(QMainWindow):
    """Main GUI window for RLM Translator"""
    
    def __init__(self):
        super().__init__()
        self.translator: Optional[RLMTranslator] = None
        self.worker: Optional[TranslationWorker] = None
        self.current_file: Optional[Path] = None
        
        self.init_ui()
        self.init_translator()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("RLM-Trans - 재귀 언어 모델 번역기")
        self.setMinimumSize(900, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Language selection
        lang_group = QGroupBox("언어 설정")
        lang_layout = QFormLayout()
        
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["자동 감지", "한국어", "일본어", "영어"])
        lang_layout.addRow("원본:", self.source_lang_combo)
        
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["한국어", "일본어", "영어"])
        self.target_lang_combo.setCurrentIndex(0)
        lang_layout.addRow("대상:", self.target_lang_combo)
        
        lang_group.setLayout(lang_layout)
        controls_layout.addWidget(lang_group)
        
        # LLM Settings
        llm_group = QGroupBox("LLM 설정")
        llm_layout = QFormLayout()
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["LM Studio", "OpenAI", "Gemini"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        llm_layout.addRow("프로바이더:", self.provider_combo)
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        llm_layout.addRow("모델:", self.model_combo)
        
        self.test_btn = QPushButton("연결 테스트")
        self.test_btn.clicked.connect(self.test_connection)
        llm_layout.addRow("", self.test_btn)
        
        llm_group.setLayout(llm_layout)
        controls_layout.addWidget(llm_group)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Text areas with splitter
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
        source_layout.addLayout(source_header)
        
        self.source_text = QTextEdit()
        self.source_text.setFont(QFont("Malgun Gothic", 11))
        self.source_text.setPlaceholderText("번역할 텍스트를 입력하거나 파일을 불러오세요...")
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
        self.target_text.setPlaceholderText("번역 결과가 여기에 표시됩니다...")
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
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.translate_btn.clicked.connect(self.start_translation)
        button_layout.addWidget(self.translate_btn)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setMinimumWidth(80)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_translation)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("준비됨")
        
        # Menu bar
        self.create_menu()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
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
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("도움말")
        
        about_action = QAction("정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_translator(self):
        """Initialize the translator"""
        try:
            config = LLMConfig.from_env()
            self.translator = RLMTranslator(llm_config=config)
            
            # Set provider combo based on config
            provider_map = {"lmstudio": 0, "openai": 1, "gemini": 2}
            self.provider_combo.setCurrentIndex(provider_map.get(config.provider, 0))
            
            # Load available models
            self.refresh_models()
            
        except Exception as e:
            self.status_bar.showMessage(f"초기화 오류: {e}")
    
    def on_provider_changed(self, provider_name: str):
        """Handle provider change"""
        provider_map = {"LM Studio": "lmstudio", "OpenAI": "openai", "Gemini": "gemini"}
        provider = provider_map.get(provider_name, "lmstudio")
        
        # Recreate translator with new provider
        config = LLMConfig.from_env()
        config.provider = provider
        
        try:
            self.translator = RLMTranslator(llm_config=config)
            self.refresh_models()
            self.status_bar.showMessage(f"{provider_name} 프로바이더로 변경됨")
        except Exception as e:
            self.status_bar.showMessage(f"프로바이더 변경 실패: {e}")
    
    def refresh_models(self):
        """Refresh available models"""
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
        """Test LLM connection"""
        if not self.translator:
            QMessageBox.warning(self, "오류", "번역기가 초기화되지 않았습니다.")
            return
        
        self.status_bar.showMessage("연결 테스트 중...")
        
        if self.translator.test_connection():
            QMessageBox.information(self, "성공", "LLM 서버에 연결되었습니다.")
            self.refresh_models()
            self.status_bar.showMessage("연결 성공")
        else:
            QMessageBox.warning(self, "실패", "LLM 서버에 연결할 수 없습니다.\n설정을 확인해주세요.")
            self.status_bar.showMessage("연결 실패")
    
    def load_file(self):
        """Load text file"""
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
                self.status_bar.showMessage(f"파일 로드됨: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"파일을 열 수 없습니다: {e}")
    
    def save_file(self):
        """Save translation result"""
        if not self.target_text.toPlainText():
            return
        
        # Suggest file name
        suggested = ""
        if self.current_file:
            target_lang = self._get_target_lang_code()
            suggested = str(self.current_file.parent / f"{self.current_file.stem}_{target_lang}{self.current_file.suffix}")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "저장", suggested,
            "텍스트 파일 (*.txt);;SRT 자막 (*.srt);;모든 파일 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.target_text.toPlainText())
                self.status_bar.showMessage(f"저장됨: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"저장 실패: {e}")
    
    def copy_result(self):
        """Copy result to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.target_text.toPlainText())
        self.status_bar.showMessage("클립보드에 복사됨")
    
    def _get_source_lang_code(self) -> str:
        """Get source language code from combo"""
        lang_map = {"자동 감지": "auto", "한국어": "ko", "일본어": "ja", "영어": "en"}
        return lang_map.get(self.source_lang_combo.currentText(), "auto")
    
    def _get_target_lang_code(self) -> str:
        """Get target language code from combo"""
        lang_map = {"한국어": "ko", "일본어": "ja", "영어": "en"}
        return lang_map.get(self.target_lang_combo.currentText(), "ko")
    
    def start_translation(self):
        """Start translation"""
        text = self.source_text.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "오류", "번역할 텍스트를 입력해주세요.")
            return
        
        if not self.translator:
            QMessageBox.warning(self, "오류", "번역기가 초기화되지 않았습니다.")
            return
        
        # Reset cost tracking
        self.translator.reset_costs()
        
        # UI state
        self.translate_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.target_text.clear()
        
        # Create worker thread
        self.worker = TranslationWorker(
            self.translator,
            text,
            self._get_source_lang_code(),
            self._get_target_lang_code()
        )
        
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_translation_finished)
        self.worker.error.connect(self.on_translation_error)
        
        self.worker.start()
    
    def cancel_translation(self):
        """Cancel ongoing translation"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.on_translation_finished(None)
            self.status_bar.showMessage("번역 취소됨")
    
    def on_progress(self, message: str, progress: float):
        """Handle progress update"""
        self.progress_bar.setValue(int(progress * 100))
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)
    
    def on_translation_finished(self, result: Optional[TranslationResult]):
        """Handle translation completion"""
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
                f"번역 완료 - {result.chunks_count}개 청크, "
                f"호출 {cost['total_calls']}회"
            )
        else:
            if result.translated_text:
                self.target_text.setPlainText(result.translated_text)
            QMessageBox.warning(self, "오류", f"번역 중 오류 발생: {result.error_message}")
    
    def on_translation_error(self, error: str):
        """Handle translation error"""
        self.translate_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "오류", f"번역 실패: {error}")
        self.status_bar.showMessage("번역 실패")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "RLM-Trans 정보",
            "RLM-Trans v1.0\n\n"
            "재귀 언어 모델(RLM) 기반 번역기\n"
            "한국어, 일본어, 영어 번역 지원\n\n"
            "긴 문서도 컨텍스트를 유지하며 번역합니다."
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = RLMTranslatorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
