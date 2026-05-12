# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드를 다룰 때 참고하는 안내서입니다.

## 프로젝트 개요

심부 콘크리트 보(Deep Beam)에 대한 스트럿-타이 모델(STM)을 자동 생성하는 멀티 에이전트 시스템(MAS). SIMP 위상최적화, 로컬 LLM 에이전트(Ollama), KDS 14 20 24 기반 코드 검증을 결합하여 사용합니다.

## 실행 방법

**사전 요구사항:** Ollama가 `http://localhost:11434`에서 실행 중이어야 하며, `deepseek-r1:14b` 및 `qwen2.5:7b` (또는 `qwen3:32b`) 모델이 필요합니다.

```bash
pip install -r requirements.txt   # numpy, matplotlib, anthropic (anthropic은 목록에 있으나 실제로는 Ollama 사용)
pip install scipy                 # cdr_calc.py에 필요
```

**메인 실행** — `mas_stm_final.py`:
```python
from mas_stm_final import MASSTMGenerator
mas = MASSTMGenerator(topology_model="deepseek-r1:14b", reviewer_model="qwen2.5:7b",
                      max_retries=5, n_candidates=3, output_dir="plots_8node")
result = mas.generate(beam_length=6900, beam_height=2000, beam_width=500,
                      loads=[(2225, -2000), (4675, -2000)],
                      support_positions=[(225, 'pin'), (6675, 'roller')], verbose=True)
```

**절점 영역 검증** — `nodal_zone_design.py`:
```python
from nodal_zone_design import design_all_candidates
result = design_all_candidates(json_path='stm_result.json', fck=27, fy=400, bw=500,
                                bearing_plate=450, cover=40, stirrup_dia=16, main_bar_dia=32,
                                phi=0.75, save_dir='plots_results', verbose=True)
```

## 아키텍처

`MASSTMGenerator.generate()` 내부에서 순차적으로 실행되는 파이프라인:

1. **Phase 0 — SIMP 위상최적화** (`simp_optimize()`): 밀도 기반 최적화로 보 내부의 자연스러운 힘의 경로를 식별. 파라미터: `volfrac=0.3`, `penal=3.0`, `rmin=1.5`.

2. **Phase 1A — LLM 절점 배치** (`_decide_node_layout()`): LLM이 SIMP 결과를 해석하여 중간 절점 쌍 개수(0/1/2)와 위치를 결정. `OllamaClient`의 `force_json=True`로 구조화된 출력 사용.

3. **Phase 1B — 그라운드 스트럭처 생성** (`enumerate_ground_structure()`): 유효한 모든 부재 연결을 결정론적으로 열거. 제약조건: 각도 25-65도, 교차 불가.

4. **Phase 2 — 공학적 검토** (`_engineering_review()`): LLM 리뷰어가 하중 경로 연속성, 힘의 흐름 각도, 대칭성, 완전성을 평가.

5. **Phase 3 — 트러스 해석** (`solve_truss()`): 직접강성법으로 2D 트러스 해석. 평가 지표: `Sum(|인장력| x 부재길이)` — 낮을수록 효율적.

6. **Phase 4 — 절점 영역 설계** (`nodal_zone_design.py`): KDS 14 20 24에 따른 스트럿/타이 폭 검증, 절점 유형 분류(CCC/CCT/CTT).

## 주요 데이터 구조

```python
# STM 위상
stm = {
    'nodes': {'A': (x, y), ...},              # 절점ID -> (x_mm, y_mm)
    'connections': [['A', 'B'], ...],          # 부재 쌍
    'supports': {'A': 'pin', 'F': 'roller'},   # 지점 조건
}

# 해석 결과
result = {
    'member_forces': {('A', 'B'): force_N, ...},  # + 인장, - 압축
    'reactions': {'A': (Rx_N, Ry_N), ...},         # 반력
}
```

결과는 `stm_result.json`에 전체 후보군과 최적 STM을 포함하여 저장됩니다.

## 주요 파일

| 파일 | 용도 |
|------|------|
| `mas_stm_final.py` | 메인 파이프라인: SIMP + LLM 에이전트 + 트러스 해석 |
| `mas_stm_cooper_ver6_4.py` | 문법 기반 규칙 선택 변형 버전 |
| `nodal_zone_design.py` | KDS 14 20 24 절점 영역 검증 및 시각화 |
| `cdr_calc.py` | 스트럿/타이 폭 계산 (Nelder-Mead 최적화) |
| `archive/` | 이전 구현 버전들 |

## 참고 사항

코드 주석과 커밋 메시지는 한국어로 작성. 한국 구조공학 기준(KDS)을 따르며, 단위는 mm/kN/MPa를 사용합니다.
