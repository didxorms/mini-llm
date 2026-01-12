# mini-llm
Tiny character-level Transformer LM (from scratch).

# Mini-LLM Notes (Day 6)

이 프로젝트를 하며 생긴 궁금증을 정리한 노트.

---

## 1) argparse는 왜 쓰는가?

코드를 수정하지 않고 실행 옵션만 바꿔서 실험을 반복/재현하기 위해 쓴다.

- `--seq_len`, `--max_steps`, `--temp`, `--top_k`, `--cache` 같은 옵션으로 실험 설정을 빠르게 바꿀 수 있음
- GitHub에 올렸을 때 다른 사람도 같은 커맨드로 결과를 재현 가능
- 여러 실험을 자동으로 돌리기(스크립트/배치 실행) 쉬움

---

## 2) `nn.Module`은 뭐고 왜 상속하나?

PyTorch에서 “학습 가능한 레이어/모델”의 표준 베이스 클래스.

`nn.Module`을 상속하면 자동으로 얻는 것:
- 내부 레이어(`nn.Linear`, `nn.Embedding` 등) 파라미터가 자동 등록됨 (`model.parameters()` 가능)
- `model.to(device)`, `model.eval()`/`model.train()`이 하위 모듈까지 적용됨
- `state_dict()`로 저장/로드 가능
- Dropout 같은 레이어가 train/eval에 따라 자동으로 동작이 바뀜

그래서 레이어/블록/모델 만들 때 대부분 `nn.Module`을 상속한다.

---

## 3) top-k는 무엇인가?

샘플링(생성) 시 “다음 토큰 후보를 확률 상위 k개로 제한”하는 기법.

- 모델이 다음 토큰 확률 분포 `p(token | context)`를 만들고
- 상위 k개만 남긴 뒤 그 안에서 샘플링

효과:
- 말도 안 되는 희귀 토큰이 튀는 걸 줄임(난삽함 감소)
- greedy(항상 1등)보다 반복이 줄고 다양성이 생김

---

## 4) `.pt` 파일에는 무엇이 저장되나?

보통 `torch.save(dict, path)`로 저장하는 “체크포인트” 파일.

프로젝트 기준으로 대략 이런 dict가 들어간다:
- `"model"`: `model.state_dict()` (가중치 텐서들)
- `"optim"`: optimizer 상태(모멘텀 등) (저장하는 경우)
- `"step"`: 학습 스텝
- `"max_len"`, `"vocab_size"`, `"tokenizer"`: 실행/복원용 메타데이터

즉 `.pt`는 “모델 파라미터 + 학습 상태/메타정보”가 들어있는 파일이다.

---

## 5) lr / weight_decay / dropout은 무엇인가?

### lr (learning rate)
- 한 번 업데이트할 때 가중치를 얼마나 크게 바꿀지 결정
- 너무 크면 발산, 너무 작으면 학습이 느림

### weight_decay (정규화)
- 가중치가 너무 커지지 않도록 압박하는 정규화(AdamW에서 흔히 사용)
- 과적합 완화/일반화에 도움

### dropout
- 학습 중 일부 뉴런 출력을 랜덤으로 0으로 만들어 과적합을 줄이는 기법
- train에서만 적용, eval에서는 꺼짐

---

## 6) mask는 왜 쓰는가?

언어모델(다음 토큰 예측)은 미래 토큰을 보면 반칙이므로,
현재 위치 t가 `<= t`까지만 보도록 막는 **causal mask**가 필요하다.

구현은 보통 `torch.tril(...)`로 “삼각형(과거만 허용)” 마스크를 만든다.

---

## 7) 토크나이저도 학습시켜야 하는 거 아님? (유사 단어 벡터 관련)

중요한 구분:

- 토크나이저는 “문자열을 토큰 ID로 쪼개는 규칙”이다.
- “유사 단어가 비슷한 벡터”가 되는 건 토크나이저가 아니라 모델의 `nn.Embedding`이 학습되면서 생긴다.
  (비슷한 문맥에서 쓰이는 토큰들이 비슷한 임베딩을 갖게 됨)

다만 토크나이저 자체도 보통 “학습”하는 방식이 많다:
- BPE/SentencePiece처럼 데이터에서 자주 나오는 subword 조각을 자동으로 만들기 위해 학습함
- 이건 “의미 벡터 학습”이 아니라 “어떤 조각 단위로 쪼갤지”를 학습하는 것

현재 프로젝트는 byte-level 토크나이저(0~255)라 토크나이저 학습이 없다.

---

## 8) Causal attention이 뭐고, attention / cross-attention과 차이?

- Attention: Q,K,V를 이용해 가중합하는 일반 메커니즘
- Self-attention: Q,K,V가 같은 시퀀스에서 나옴
- Cross-attention: Q는 디코더에서, K/V는 다른 시퀀스(인코더 출력)에서 나옴 (encoder-decoder에서 사용)
- Causal self-attention: self-attention이지만 미래를 못 보게 마스크를 적용한 것 (GPT류 생성 모델에서 사용)

---

## 9) KV-cache를 하면 왜 출력이 빨라짐?

생성은 “한 토큰씩” 이어서 만든다.

### cache OFF (느림)
매 토큰마다 문맥 전체 길이 T에 대해 Q,K,V를 다시 계산하고,
attention도 전체 문맥에 대해 다시 수행한다.
→ 문맥이 길어질수록 매 스텝 비용이 커짐.

### cache ON (빠름)
prefill 단계에서 과거 토큰들의 K,V를 한 번 계산해 저장해두고,
이후에는 새 토큰 1개에 대한 K,V만 추가하며 과거 K,V는 재사용한다.
→ “전체 재계산”이 “새 토큰만 계산”으로 바뀌어 크게 빨라진다.

(실제로 벤치마크에서 cache ON이 cache OFF보다 큰 폭으로 tokens/s가 증가함)
