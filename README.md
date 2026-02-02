# RLM-Trans: 재귀 언어 모델 기반 번역기

긴 문서도 **컨텍스트를 유지**하면서 번역하는 한/일/영 번역기입니다.

## 특징

- 🔄 **재귀 언어 모델(RLM)** 기반으로 긴 문서도 맥락을 유지하며 번역
- 🌏 **한국어/일본어/영어** 상호 번역 지원
- 🎬 **SRT 자막** 파일 번역 지원
- 🖥️ **로컬 LLM** (LM Studio) 및 **클라우드 API** (OpenAI, Gemini) 지원
- 📊 용어집(Glossary) 기반 일관된 번역

## 설치

```bash
cd f:\antig\RLM-trans
pip install -r requirements.txt
```

## 설정

`.env.example`을 `.env`로 복사하고 API 키를 입력하세요:

```bash
copy .env.example .env
```

### 프로바이더별 설정

**LM Studio (기본값):**
```env
DEFAULT_PROVIDER=lmstudio
LM_STUDIO_URL=http://localhost:1234/v1
```

**OpenAI:**
```env
DEFAULT_PROVIDER=openai
OPENAI_API_KEY=sk-xxx...
```

**Google Gemini:**
```env
DEFAULT_PROVIDER=gemini
GEMINI_API_KEY=xxx...
```

## 사용법

### GUI 실행
```bash
python translator_gui.py
```

### CLI 테스트
```bash
python rlm_translator.py
```

### 코드에서 사용

```python
from rlm_translator import RLMTranslator

translator = RLMTranslator()

result = translator.translate(
    text="번역할 텍스트",
    source_lang="ko",  # auto, ko, ja, en
    target_lang="en"
)

print(result.translated_text)
print(f"비용: {result.cost_summary}")
```

## 작동 원리

```
┌─────────────────────────────────────────┐
│          원본 텍스트 입력                │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│    REPL 환경에 컨텍스트 저장             │
│    - 원문, 용어집, 맥락 요약             │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│    주 에이전트: 청크 분할                │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│    서브 에이전트: 청크별 번역            │
│    (이전 맥락 + 용어집 참조)             │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│          최종 번역문 출력                │
└─────────────────────────────────────────┘
```

## 파일 구조

```
RLM-trans/
├── rlm_translator.py      # 메인 번역 엔진
├── repl_environment.py    # REPL 환경 (컨텍스트 저장)
├── llm_client.py          # 멀티 프로바이더 LLM 클라이언트
├── prompts.py             # 시스템 프롬프트
├── text_utils.py          # 텍스트 유틸리티
├── config.py              # 설정 관리
├── translator_gui.py      # PyQt6 GUI
├── requirements.txt       # 의존성
└── .env.example           # API 키 템플릿
```

## 라이선스

MIT License
