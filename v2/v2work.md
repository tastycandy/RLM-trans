현재 만들어 진 번역기에 대한 개선을 할거야. 이전의 readme를 읽고 참고해
그리고 다음 목표를 달성해야되


0) 목표 정의
현재 번역기는 “청킹 → 순차 번역” 중심이야. 여기서 RLM(루트가 REPL을 통해 도구를 쓰며 반복적으로 계획/조회/수정)을 붙이는 목표는 이거야:

문서 전체에서 일관성 유지 (용어/인물/말투/문체)
실패를 자동 감지하고 부분 재시도 (형식 깨짐, 누락, 길이 폭주, 금지어 포함 등)
필요한 정보만 조회해 최소 토큰으로 유지 (긴 문서에서도 효율)
다양한 작업 타입(자막/특허/논문/일반)을 프리셋으로 안정화
1) 핵심 구성요소(역할 분리)
A. Root Orchestrator (루트 에이전트)
문서 전체를 책임지는 “감독”
매 라운드마다:
현재 상태(진행률/용어집/스타일 가이드/최근 오류)를 읽고
“다음에 무엇을 번역할지/어떻게 번역할지”를 계획하고
필요한 도구 호출(REPL)을 지시하고
sub-agent 번역 결과를 검증하고
상태를 갱신함
B. Sub Translator (서브 번역 에이전트)
“번역만 잘 하는” 작업자
입력: (원문 chunk) + (루트가 만든 컨텍스트 패키지)
출력: (번역 결과) + (발견한 용어 후보/주의사항)
C. Verifier / Critic (검증기)
번역 결과가 룰을 만족하는지 검사
검증은 “규칙 기반 + 짧은 LLM 검증” 혼합 권장
규칙 기반: SRT 줄 수/인덱스 유지, 금지문자, delimiter 누락, 길이 제한 등
LLM 검증: 의미 보존/자연스러움/톤/용어 일관성 체크
D. REPL Environment (도구 상자)
루트가 코드/명령으로 호출할 수 있는 “안전한” 도구 API:

get_chunk(i) / peek_chunks(range)
extract_terms(text) / update_glossary(pairs)
summarize_context(last_k_translations)
check_constraints(translation, preset_rules)
compute_similarity(prev, cur) (예: 톤/길이 편차 감지)
save_translation(i, text) / get_translation(i)
select_next_chunk(strategy) (동적 스케줄링)
여기서 중요한 건 “exec로 파이썬 마음대로 실행”이 아니라, 화이트리스트된 함수만 호출 가능한 DSL/함수콜 형태로 제한하는 것(안전).
2) 상태(State) 설계: “번역 프로젝트 메모리”
문서가 길어질수록 “무슨 결정이 내려졌는지”가 제일 중요해져.

TranslationState에 들어갈 것
preset_id (자막/특허/논문/일반)
chunk_plan (chunk 경계, overlap 여부, chunk별 우선순위)
glossary (용어→번역 매핑 + 확신도/출처 chunk)
style_guide (말투/문체/존댓말/문장 길이/번역 금지 규칙 등)
entities (인물/지명/조직 등 표기 통일)
history_summaries (최근 N개 chunk의 요약/결정사항)
quality_flags (오류/재시도 횟수/문제 유형)
cost_stats (루트/서브/검증 비용과 호출수)
이 상태는 “매 라운드마다 업데이트”되어야 하고, 루트는 상태를 근거로 행동해야 RLM 느낌이 살아.
3) 실행 루프(진짜 RLM의 핵심)
Round-based loop (예시)
각 라운드는 아래 6단계를 돈다:

Plan
루트가 state + 진행상황을 보고
다음 chunk 선택 + 필요한 컨텍스트 생성 전략 결정
예: “용어가 많이 등장하는 구간 먼저”, “대화 장면은 톤 고정 우선”, “특허는 청구항 룰 강제”
Retrieve
REPL로 필요한 것만 가져옴
get_chunk(i) + 필요시 peek_chunks(i-1,i+1)
glossary/style_guide 요약본 생성(짧게)
Translate
서브 번역 모델 호출
입력 패키지:
원문 chunk
필수 규칙(프리셋)
glossary 강제(“반드시 이 번역 사용”)
최근 문맥 요약(짧게)
금지사항/형식 템플릿
Verify
규칙 기반 검사(형식/구조/누락)
LLM 검증(필요할 때만)
실패면 “왜 실패했는지”를 구조화된 에러로 기록
Repair (조건부)
실패 유형별 자동 복구 루틴
형식 오류 → 템플릿 재강제 + 재번역
용어 불일치 → glossary 업데이트 후 재번역
문장 너무 길다 → 압축/분할 재요청
의미 누락 → 누락된 문장만 재번역
Commit
save_translation(i, final_text)
새로운 용어 후보를 glossary에 “후보”로 등록
context_summary 갱신
다음 라운드로
종료 조건:

모든 chunk 번역 완료
또는 품질/비용/재시도 제한 도달(그럴 땐 “문제 chunk 목록 + 이유”를 결과로 함께 출력)
4) “컨텍스트 패키지” 설계(서브 입력을 깔끔하게)
서브에게 주는 정보는 길면 망하고, 짧으면 일관성 망가져. 그래서 고정 포맷으로:

Rules: 프리셋 핵심 규칙(최대 10줄)
Glossary (hard): 반드시 지킬 용어 20~50개(상위 중요도)
Style: 문체/말투/포맷 지침
Local context: 직전 chunk 요약 3~5줄 + 등장인물 표기
Chunk: 현재 번역할 원문
이 패키지는 REPL에서 자동으로 생성되게 하고, 루트는 “얼마나 넣을지”만 결정하는 식이 좋아.
5) 청킹 전략(단순 n줄이 아니라 “의미 기반 + 오버랩”)
RLM 번역기 품질은 청킹이 반 이상을 결정해.

기본은 “문단/문장 경계” 우선(가능하면)
자막(SRT)은 엔트리 단위 유지가 최우선
특허/청구항은 문장 하나로 유지 같은 하드 룰이 있으니 청킹도 그 룰을 존중
오버랩(앞뒤 1~3문장 또는 150~300자)을 옵션으로 두고, overlap은 번역 결과에 중복 삽입되지 않도록 “참조용”으로만 사용
6) 용어집(Glossary) 운영 로직
용어집은 “자동 추출”보다 “자동 확정/강제”가 핵심.

서브가 번역 결과와 함께 term_candidates를 출력
루트는 후보를 보고:
이미 있는 항목이면 충돌 여부 검사
충돌이면 “우선순위 규칙” 적용:
프리셋/사용자 고정 > 문서 초반 정의 > 다수결 > 최신
확정된 용어는 다음 chunk들에서 “hard glossary”로 강제
추가로:

“고유명사/제품명/인명”은 별도 entities 테이블로 분리하면 더 안정적
7) 품질/안전 장치(최소 세트)
품질 체크(규칙 기반)
SRT: 인덱스/시간코드 보존, 줄 수/분리자 보존, 비어있는 라인 방지
일반: 누락 감지(원문 대비 문장 수 급감), 금지문자/금지어
특허: “단일문장/세미콜론/wherein 1회” 같은 룰 체크
실행 안전
“LLM이 만든 코드 exec” 금지(또는 강력 샌드박스)
REPL은 화이트리스트 함수만 호출 가능하게 설계
로그에 원문 전체를 남기지 않거나(민감정보) 옵션화
8) 비용/속도 최적화 운영
루트 모델은 “작고 빠른 모델” + 필요 시만 큰 모델로 승격
Verifier도 기본은 규칙 기반, “문제 징후 있을 때만” LLM 검증
chunk 실패 재시도는 “부분 재번역” 우선(전체 재번역 금지)
9) 단계별 개발 로드맵(구현 없이 계획만)
Phase 1: 상태/루프만 붙이기(가장 작은 RLM)
TranslationState 도입
“Plan→Retrieve→Translate→Verify→Commit” 1회전 루프
REPL은 get_chunk, save_translation, summarize_context, check_constraints만
Phase 2: Repair + Glossary
실패 유형별 repair 루틴
용어 후보 추출/확정/강제 흐름 추가
Phase 3: 동적 청킹/스케줄링
어려운 chunk 우선 처리(예: 특허 청구항/대화 톤)
overlap/semantic chunking 도입
Phase 4: 평가/리그레션
자막 5개, 특허 5개, 논문 5개 샘플 세트
품질 지표:
형식 성공률
재시도율
용어 일관성 점수(간단히: glossary 위반 횟수)
비용/속도



 ## 그리고 이건 개선을 위한 구체적플랜이야 이것을 수행하기 위한 구체적인 계획을 짜고 코딩을 하면 되
  
목표
일관성(용어/톤/표기) 유지
자동 검증 + 자동 복구(재시도) 로 실패율 낮추기
필요한 정보만 조회해서 긴 문서도 효율적으로 처리
**프리셋(자막/특허/논문/일반)**으로 규칙을 안정화
핵심 컴포넌트(역할 분리)
Root Orchestrator(루트)
전체 진행/계획/컨텍스트 생성/검증 결과 반영/상태 갱신 담당
Sub Translator(서브)
“번역만” 담당
입력: chunk + 컨텍스트 패키지
출력: 번역 + 용어 후보 + 주의사항
Verifier/Critic(검증기)
규칙 기반 검사(형식/누락/금지어)
필요 시만 짧은 LLM 검증(의미/자연스러움/일관성)
REPL Environment(도구 상자)
루트가 호출하는 화이트리스트 함수 모음
예: get_chunk, peek_chunks, summarize_context, extract_terms, update_glossary, check_constraints, save_translation
핵심 데이터: TranslationState(프로젝트 메모리)
preset_id
chunk_plan(경계/overlap/우선순위)
glossary(용어→번역, 확신도/출처)
style_guide(문체/존댓말/길이/금지 규칙)
entities(인명/지명/조직 표기 통일)
history_summaries(최근 요약/결정사항)
quality_flags(오류 유형/재시도 횟수)
cost_stats(루트/서브/검증 비용)
실행 루프(진짜 RLM 핵심)
각 라운드:

Plan: 다음 chunk 선택 + 전략 결정
Retrieve: REPL로 필요한 부분만 조회(주변 chunk/요약/용어)
Translate: 서브 번역 호출(컨텍스트 패키지 포함)
Verify: 규칙 검사 + 필요 시 LLM 검증
Repair(조건부): 실패 유형별 부분 재번역/수정
Commit: 저장 + glossary/style/state 갱신

종료: 전 chunk 완료 or 재시도/비용 제한 도달(문제 chunk 목록 출력)
서브에 주는 “컨텍스트 패키지” 고정 포맷
Rules: 프리셋 핵심 규칙(짧게)
Glossary(hard): 반드시 지킬 용어 상위 N개
Style: 문체/톤 지침
Local context: 직전 요약 + 인물/표기
Chunk: 현재 원문
청킹 전략
단순 길이 기준보다 문장/문단 경계 우선
필요 시 overlap은 참조용으로만(중복 삽입 금지)
SRT는 엔트리 단위 유지, 특허는 청구항 규칙에 맞춰 청킹
용어집 운영 로직
서브가 term_candidates 제시
루트가 충돌/우선순위 규칙으로 확정
확정 용어는 이후 chunk에서 hard glossary로 강제
품질/안전 장치
형식 검증(특허 단일문장/세미콜론, SRT 인덱스/타임코드 등)
실패 시 전체가 아닌 부분 repair 우선
REPL은 exec 금지/샌드박스, 화이트리스트 함수 호출만
개발 로드맵(단계)
Phase 1: State + 루프(Plan→Retrieve→Translate→Verify→Commit) 최소 구현
Phase 2: Repair + Glossary 강제
Phase 3: 동적 스케줄링 + semantic chunking/overlap
Phase 4: 평가셋으로 리그레션(형식 성공률/재시도율/용어 위반/비용)

추가
① Root의 '상태 요약' 한계 설정
문서가 길어지면 TranslationState 자체의 크기가 커져 Root의 컨텍스트 윈도우를 압박할 수 있습니다.

해결책: history_summaries를 저장할 때, 전체 이력이 아닌 '최근 3~5개 청크의 상세 이력 + 그 이전은 압축된 요약본' 형태로 슬라이딩 윈도우 방식을 적용하는 것이 효율적입니다.

② Verifier의 'Soft vs Hard' 구분
모든 에러를 재번역(Repair)하면 속도가 너무 느려집니다.

Hard Error: SRT 인덱스 파손, 특허 청구항 번호 누락, 금지어 포함 → 반드시 재번역.

Soft Error: 말투가 약간 어색함, 문장이 조금 김 → 로그만 남기거나 다음 청크에서 Root가 보정 지시.

③ Glossary의 '충돌 해결' 로직
문서 전반부와 후반부에서 LLM이 추출한 용어가 충돌할 경우, 플랜에 언급하신 '우선순위 규칙'을 REPL의 update_glossary 함수 안에 **결정론적 알고리즘(Deterministic Logic)**으로 박아두는 것이 좋습니다. (LLM의 판단에만 맡기지 마세요.)