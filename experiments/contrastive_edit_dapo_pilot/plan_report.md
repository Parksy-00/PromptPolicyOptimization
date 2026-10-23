# DAPO candidate-first likelihood refinement — 실행 전 검수

**상태: 2026-10-19 사용자가 “이대로 run을 한번 local에서 돌려줘”로 현재 조건의 로컬 실행을 승인했다.**
실행 ID: `20261019T012552Z-contrastive-edit-dapo-pilot-gpu1`. 이하 검수 대기/실행 전 표현은 승인 이전 설계 기록이다. 과학적 조건은 변경하지 않았다.
2026-10-19 사용자가 출력 규칙 `Answer: <integer>` 고정과 instruction만 최적화하는 범위를 확정했다.
이 결정을 반영했으며 나머지 연구 조건은 검수 대기다.

## 질문과 결과물

고정된 같은 문제의 성공·실패 응답에 대한 likelihood 이동을 관측한 LLM이,
실패를 더 강화한 편집을 유리한 편집으로 refine할 수 있는가?
첫 pilot의 목적은 pipeline 작동과 구체 사례 확인이다. 일반적인 성능 개선을 입증하는 규모는 아니다.

각 제안의 success gain, failure gain, margin, sum/mean 차이, mixed coverage, ablation 효과,
실제 채택 여부, validation/test, 생성·teacher-forcing·optimizer 비용을 기록한다.
원인 가설은 대응 patch와 실제 before/after 수치를 함께 보여주며 인과관계 증명으로 표시하지 않는다.

## 입력과 과거 실행의 관계

데이터 출처는 정확히 historical run **`20260917T062343Z-dapo-textgrad-ctx40-bbd2`**의 archived split이다.
기존 export인 `artifacts/inputs/textgrad_dapo_likelihood_pilot_seed0/manifest.json`을 직접 확인했다.
이름이나 비슷한 설정으로 source run을 추정하지 않았다.

- 각 원본 split의 앞 6개, reward를 보지 않은 기존 export를 재사용한다.
- Train SHA256: `a5f0ce7cde3eade2cb961fbdca55c83114c480423a72029f98d5d56a8f044a5c`.
- Validation SHA256: `f00731221f959c29aa494969dbf545d14663653a4cd41f4c5affc977dae1f703`.
- Test SHA256: `8a43267cc93eb6d74c8b3c07befc18c3e3f25413f9f373209fe2fae267512ca4`.
- Initial prompt: `Solve the problem and provide the answer in the correct format.`
- 기존 online pilot `20261016-textgrad-dapo-pilot-likelihood-ascent-gpu1`은 구현·조건 비교 맥락이다.
  Native optimizer state나 기존 응답을 이어받지 않는다. 부모 응답은 모두 새로 생성한다.
- Optimizer evidence는 train만 사용한다. Validation 출력은 inner optimizer에 넣지 않고 test는 최종 선택 후 사용한다.

독립 프로젝트에 PromptConsolidation의 stage catalog / plan_experiment.py를 도입하지 않는다.
기존 독립 CLI와 standalone TOML을 확장한다. Runner/scheduler/DB나 다른 프로젝트 import는 추가하지 않는다.

## 재사용과 새 구현의 대응

| 처리 | 구현 |
|---|---|
| Dataset identity / integer score / fixed Answer rule | `benchmarks.py`의 `answer_integer_v1` 경로 |
| Qwen 생성, thinking/EOS, context 상한 | 기존 `LocalHFBackend`; 진단용 token decode만 추가 |
| 응답 생성과 평가 | 기존 `TaskAdapter.evaluate` 재사용 |
| 정확한 IDs의 teacher forcing / cache | 기존 `TrajectoryScorer` |
| 같은 문제 contrast, sum/mean, question 평균 | `objective.py::fixed_contrastive_gain` 추가 |
| 후보 생성→측정→refine/ablate→채택 | `edit_optimizer.py` 추가 |
| 실제 system prompt와 JSON request | `edit_prompts.py` 추가 |
| Provenance, 선택 고정, 최종 test | 기존 CLI 확장 |

## 검수할 명시적 조건

Standalone source는 `experiment.toml`이다. 선택의 의미와 대안은 `docs/contrastive-edit-refinement.md`에 있다.

| 항목 | 제안값 |
|---|---|
| 모델 | Qwen3-8B, task/proposer/refiner/scoring에서 같은 고정 가중치 |
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 실행안 | local physical GPU1만 사용; TOML은 노출된 device 안의 cuda:0 |
| 정밀도 / context | BF16 / 40,960 tokens, serial inference |
| Task sampling | T=1, top-p=1, top-k=0, thinking 켬, 최대8,192 생성tokens |
| Optimizer sampling | T=0.6, top-p=0.95, top-k=0, 최대8,192 생성tokens |
| Likelihood | full vocabulary T=1, 원래 EOS를 포함하는 response IDs |
| Train/validation/test | 6 / 6 / 6 questions |
| Train batch / responses | 1 question / 2 independent rollouts |
| Outer steps | 2 |
| Inner search | initial candidates2 × 각각2 refinement rounds |
| Ablation | 각 round에서1개 측정; diagnostic만 사용 |
| 주목적 | question-balanced sequence success gain − failure gain |
| Gate | success gain > 1e-6, margin > 1e-6 |
| Validation guard | 관측 accuracy 하락0; outer step당 후보1개만 평가 |
| 채택 / 최종 선택 | gate 이후 margin 최고를 validation; 마지막에는 admitted 중 validation 최고 |
| Search RNG | 0, 문항 순서만 제어; sampling seed나 training seed를 추가하지 않음 |
| Prompt size | 편집 가능한 P 최대6,000문자, 의미적으로 한 변경을 지시, patch 일치는 코드로 검증 |
| 출력 형식 (사용자 확정) | 렌더러가 P 뒤에 고정 F를 붙임. 마지막 줄 `Answer: <integer>` |
| 정답 필터 (출력 결정의 구현) | Thinking 밖 마지막 비어 있지 않은 줄의 ASCII 정수를 추출해 gold와 정수 비교. P만 patch 대상 |
| Token windows | 128 tokens 단위, 절대 sequence delta 최대2개; 응답 본문은 생략하지 않음 |

Dry-run의 첫 train 순서: `dapo-source-14055`, `dapo-source-2084`.
Shuffle 순서이며 성공·실패 여부로 고른 것이 아니다. Mixed pair가 없으면 해당 step을 skip한다.
2응답으로 mixed pair를 얻지 못할 수 있다. 자동 추가 생성 조건은 넣지 않았다.

## 기존 TextGrad pilot과의 차이 및 이유

- Optimizer: TextGrad backward/update에서 후보 편집과 실측 refinement로 변경한다. 이번 요청의 본체다.
- Failure likelihood: 진단 대상에서 contrastive objective의 음의 항으로 바꾼다.
- Train minibatch3→1, 같은 문제의 rollouts1→2. 같은 문제에서 성공·실패를 얻으면서 전문 context를 수용한다.
- 기존2 proposals/batch에서2 initial branches ×2 rounds로 바꾼다. Ablation 측정을 다음 round에서 사용한다.
- Validation: 모든 proposal 평가에서 likelihood gate를 통과한 winner1개만 평가하는 방식으로 바꾼다.
- 문항 추출: 매번 독립 sample에서 shuffle한 train 순회로 바꾼다. 짧은 pilot의 문항 중복을 줄인다.
- 주목적: 성공만의 reward-weighted gain에서 mixed question의 성공 평균−실패 평균으로 바꾼다.
- 출력 protocol: DSPy marker wrapper 대신 `SYSTEM=P + 고정 F`, `USER=문제`를 사용한다 (사용자 요청).
- 정답 추출: DSPy의 answer 필드 대신 마지막 줄 `Answer: <integer>`를 읽는다. 정수 비교 의미는 유지하고,
  별도 reasoning 필드 존재 조건은 제거한다. 형식이 달라졌으므로 과거 reward/응답을 재사용하지 않는다.
- 위 항목 외 모델, revision, 초기 instruction P, split, task/reflection sampling, token cap은 기존 pilot과 같다.
- TextGrad source와 vendor는 수정하지 않는다. 기존 validation/test 점수를 새 실행의 측정치로 재사용하지 않는다.

출력 형식과 instruction 편집 범위는 사용자 결정이다. 그 외의 변경들은 검수 대상이며 실제 실행 승인은 아직 없다.

## 최대 예산

- Search task calls: base validation6 +2×(parent train2 + winner validation6) = **22**.
- 마지막 test: initial6 + selected6 = **최대12**. 초기 prompt가 선택되면 명시적으로 재사용하여6.
- Optimizer calls: 2 outer ×2 branches ×(propose1 + refine2) = **최대12**.
- 후보 측정 이벤트: 2 outer ×2 branches ×(proposal1 +2×[ablation1+revision1]) = **최대20**.
- Teacher-forcing cache miss 상한: outer당 parent2 + 후보10×응답2 =22, 합계 **최대44**.
  같은 rollback/후보/응답은 cache hit으로 줄어든다. TF/reflection은 task budget과 별도로 기록한다.

Search는 outer 전체의 최대 task 예산을 시작 전에 예약하며 partial validation을 하지 않는다.
Max tokens는 wall-clock 상한이 아니다. 전문 feedback이 context를 넘으면 오류를 남기고 조용히 축약하지 않는다.
작은 split의 단발 validation은 실제 성능 비하락을 보장하지 않는다.

## 실제 프롬프트와 검수 순서

1. `docs/contrastive-edit-refinement.md`의 선택 표와 실행 순서를 확인한다.
2. `src/prompt_policy/edit_prompts.py`의 proposer/refiner system 전문을 확인한다.
3. `examples/contrastive_edit/{proposer,refiner}_request.json`으로 user 입력 형태를 확인한다.
   합성 예시이며 새 방법의 실제 LLM 호출이 아니다. 사용자가 준 mean 수치는 가정값으로 유지한다.
4. `experiment.toml`의 science/model/data 조건을 확인한다.
5. 출력 형식과 instruction 편집 범위는 확정 사항으로 기록했다. 주목적 sequence/mean, ablation 채택 여부와 시점, validation 방식, rollouts/inner 횟수는 계속 검수한다.

CPU-only dry-run은 모델을 로드하거나 생성하지 않는다:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync prompt-policy experiments/contrastive_edit_dapo_pilot/experiment.toml --dry-run
```

검수 이후 실행 명령의 예시이며 아직 실행하지 않았다:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync prompt-policy experiments/contrastive_edit_dapo_pilot/experiment.toml
```

구현 테스트와 dry-run은 이번 coding 요청에 포함된다. 실제 GPU 연구 실행은 사용자가 요청한 디자인 검수와
독립 repo AGENTS.md의 explicit conditions/data/model 승인 이후에 수행한다.


## 2026-10-19 승인 후 실행 결과

Run `20261019T012552Z-contrastive-edit-dapo-pilot-gpu1` completed. 두 outer step 모두 reward [0,0]으로 mixed pair를 얻지 못해 skip됐다. 후보 생성/refinement/TF 측정/실제 업데이트 모두 0회. Validation 0/6, test 0/6; 최종 prompt는 base와 같다. 자세한 진단은 run의 summary.md와 verification.json에 기록했다. 과학적 조건은 변경하지 않았다.
