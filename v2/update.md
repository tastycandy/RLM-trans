1단계: REPL 및 상태(State) 고도화 (인프라 준비)가장 먼저 repl_environment.py와 TranslationState를 수정하여 용어를 관리할 공간과 도구를 정의해야 합니다.상태 정의: TranslationState에 glossary 외에 term_candidates(서브 에이전트가 제안한 후보군)와 confirmed_terms(사용자 또는 루트가 확정한 용어)를 구분하여 저장합니다.REPL 도구 추가:propose_terms(dict): 서브 에이전트가 발견한 새 용어를 후보군에 넣는 함수입니다.update_glossary(key, value, force=True): 루트 에이전트가 후보 중 적절한 것을 골라 정식 용어집에 등록하는 함수입니다.get_hard_glossary(): 다음 번역 시 서브 에이전트에게 강제할 '필수 용어 리스트'를 텍스트 형태로 추출합니다.2단계: 서브 에이전트(Sub-Translator) 출력 구조화서브 에이전트가 번역만 하는 것이 아니라, 번역 과정에서 발견한 명사를 보고하도록 프롬프트를 수정합니다. 대상 범위: 고유명사(인명/지명/제품명)뿐만 아니라 기술적 핵심 명사(명세서 내 정의된 용어) 및 도면 부호를 포함합니다. 출력 형식 제안: LLM이 파이썬 딕셔너리나 JSON 형태로 결과를 주도록 유도합니다.JSON{
  "translated_text": "...번역문...",
  "term_candidates": {
    "Source_Term": "Target_Term",
    "Reference_Sign_100": "제어부"
  }
}
3단계: 루트 에이전트(Root Orchestrator) 루프 설계루트 에이전트가 매 라운드마다 용어의 일관성을 검토하고 결정을 내립니다.계획(Plan): 이번 청크에서 번역할 내용과 함께, 지금까지 확정된 glossary를 불러옵니다.호출(Call): 서브 에이전트에게 **"이 용어집은 반드시 준수할 것"**이라는 지시와 함께 청크를 보냅니다. 분석(Analyze): 서브가 보낸 term_candidates를 검토합니다.충돌 발생 시: 이미 있는 용어와 다른 번역어를 제안했다면, 루트가 기존 용어의 일관성을 지킬지 새 용어로 업데이트할지 결정합니다.갱신(Commit): 확정된 용어를 REPL을 통해 TranslationState에 영구 저장합니다.4단계: '컨텍스트 패키지' 강화 (일관성 강제)서브 에이전트가 번역을 시작하기 전에 받는 정보에 용어집을 정교하게 배치합니다. Hard Glossary (필수): 문서 전체에서 통일되어야 하는 도면 부호 및 핵심 기술 용어. Soft Glossary (참고): 문맥에 따라 변할 수 있지만 권장되는 번역어. Style Guide: "상기(the)", "본 발명(the present invention)" 등 명세서 특유의 반복 문구 고정. 5단계: 관리 대상 명사 가이드라인효율적인 운영을 위해 무엇을 용어집에 넣을지 기준을 세웁니다.구분관리 대상 명사번역 방식고유명사인명, 회사명, 지명, 상표명1:1 매칭 고정 (Transliteration 포함)도면 부호100(제어부), 200(센서) 등숫자와 결합된 명사는 절대 불변기술 명사Member, Assembly, Means 등문서 내 정의된 의미로 일관성 강제일반 명사Thing, Process, Result 등문맥에 따르되 가급적 통일 (Soft 관리)

추가: 번역된것을 저장할때 같은 이름이 있는 경우 뒤에 숫자를 하나씩 증가시키면서 붙이기
     누락확인에서 길이검토를 안하는 것이 디폴트가 되도록