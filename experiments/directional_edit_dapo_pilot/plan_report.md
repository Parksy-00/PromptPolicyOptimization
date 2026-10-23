# Minimal likelihood-editing DAPO pilot

상태: 사용자가 요청한 단순화의 구현/CPU 테스트/dry-run. GPU 연구 실행은 시작하지 않았다.

비교 기준은 완료된 `20261019T012552Z-contrastive-edit-dapo-pilot-gpu1`이다. 그 run의 config/source snapshot은 보존하며
현재 코드로 과거 복잡한 조건을 묵시적으로 재해석하지 않는다. 원래 split 출처는
`20260917T062343Z-dapo-textgrad-ctx40-bbd2`의 archived first6/split export다.

## 현재 과학적 조건

- 모델: Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, context40960.
- Task/optimizer sampling: temperature0.6, top-p0.95, top-k20, cap8192, thinking on.
- 초기 instruction: `Solve the problem and provide the answer in the correct format.`
- 출력 규약: `answer_tag_integer_v2`; 최종 정수는 단일 `<answer>...</answer>` 안에서만 추출한다.
- Task system=editable instruction, user=고정 format constraint + `\n\nQuestion:\n` + 문제.
- Proposer/refiner 출력은 단일 `<instruction>...</instruction>` 안에서만 추출한다. 누락/중복/미완성은 기록 후 오류로 중단하며 전체 출력 fallback은 없다.
- Train pool6문항, minibatch1, 문항당 rollout2, outer2, search RNG0. Sampling seed는 추가하지 않는다.
- Proposer1회 + Refiner1회/step. 각 호출은 완성된 instruction만 생성한다.
- Proposer는 실패 진단/성공 재현을 위한 instruction을 제안하고, refiner는 개별 likelihood 변화를 추가로 받는다.
- 원래 IDs의 T1 full-vocab sequence logprob로 각 성공 상승/실패 하락을 검증한다. Cutoff는 부호0이다.
- 제안/refinement 중 검증된 G>0 최고를 반영한다. 없으면 부모 유지. 마지막 채택 prompt가 최종이다.
- Validation은 사용하지 않는다. Test6문항은 선택 고정 후 initial/selected 각각1회이며 같은 prompt면 재사용한다.

제거: hypothesis/원인 요소/ablation/patch/branch/history/token windows/복수 refinement/validation filter 및 selection.
기존 개별 trajectory 방향 검증은 유지한 구현 선택이다. 같은 pool 내 점수만 비교한다.
Search task cap22는 유지하지만 outer2×batch1×rollout2이므로 실제 search 생성은 최대4회다.
그 외 optimizer 최대4회, 측정 후보 최대4개, TF cache miss 최대12, test 최대12회다.
Validation 생성은0회다. 실패/형식/길이 문제도 원래 reward로 그대로 관측한다.

Train 순서는 `dapo-source-14055`, `dapo-source-2084`로 유지한다.

## 입력 identity

- train SHA256 `a5f0ce7cde3eade2cb961fbdca55c83114c480423a72029f98d5d56a8f044a5c`
- validation SHA256 `f00731221f959c29aa494969dbf545d14663653a4cd41f4c5affc977dae1f703` (미사용)
- test SHA256 `8a43267cc93eb6d74c8b3c07befc18c3e3f25413f9f373209fe2fae267512ca4`

`experiment.toml`은 inheritance 없는 standalone다. `[edit_refinement]`는 경로 선택용 빈 table이며,
`[objective]`에는 mode만 남겼다. 구현 선택과 실제 system은 docs/contrastive-edit-refinement.md 및
examples/contrastive_edit/에서 검수할 수 있다. 실제 재실행은 별도 사용자 요청 이후 새 run directory로 한다.

## 구현 검증

CPU 전체 테스트 106 passed / 1 skipped(CUDA-only), Ruff lint/format 통과.
Dry-run executable=true/missing=[]이며 step당 proposer1/refiner1, 측정2, train rollout2를 확인했다.
새 GPU 연구 실행은 수행하지 않았다.


2026-10-19 tag 경계 변경: 사용자 요청에 따라 미실행 recipe의 task renderer/scorer만
`answer_tag_integer_v1`로 바꿨다. 모델/데이터/sampling/budget는 그대로이며 새 GPU run은 하지 않았다.
완료된 vLLM run `20261019T063416Z-directional-edit-dapo-vllm-b3-gpu1`의 config와 artifact는
기존 `answer_integer_v1` 규약으로 보존한다. 현재 prompt/output 예시는 examples/contrastive_edit/에 있다.


추가 사용자 요청: format 규칙을 user 맨 앞으로 옮기고 그 다음 `Question:`으로 문제를 구분했다.
현재 미실행 recipe만 v2 메시지 구성으로 갱신했으며, 기존 실행 조건과 artifact는 유지한다.
Renderer 역할을 구분하기 위한 v2이고 정수/tag reward 판정은 v1과 같다. 추가 연구 실행 없음.
