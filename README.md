# AIoT_Teamproject
## 💾🧠 On-device training 메모리 최적화 비교
> Breaking the Memory Wall — ResNet18 + CIFAR-10 기반 메모리 최적화 프레임워크 비교 연구

**아주대학교 | 남현원, 김평주**

---

## Abstract

본 연구는 온디바이스 학습 환경에서의 메모리 최적화를 목표로, 세 가지 주요 메모리 최적화 프레임워크(Melon, Sage, Dropback)를 비교 분석하고, 이를 기반으로 개선된 프레임워크를 제안하였다.

온디바이스 학습은 데이터 생성 지점에서 직접 학습을 수행함으로써 개인화된 예측과 추천을 가능하게 하지만, 높은 메모리 사용량과 연산 비용으로 인해 제약이 따른다. 본 연구는 각 프레임워크의 메모리 효율성과 학습 성능을 ResNet18 모델과 CIFAR-10 데이터셋을 활용하여 실험적으로 평가하였다.

---

## 1. Introduction

On-device learning은 데이터 생성 지점에서 직접 학습을 수행함으로써 개인정보 보호를 강화하고, 네트워크 의존성을 줄이며, 맞춤형 예측과 추천을 가능하게 하는 중요한 기술이다.

- 클라우드 의존 방식은 과도한 컴퓨팅 자원과 전력 소비를 초래
- 민감한 데이터가 외부 서버로 전송되는 과정에서 개인정보 유출 위험 존재
- **Memory Wall**: 프로세서 성능이 빠르게 향상되는 반면, DRAM 발전 속도가 상대적으로 느려 메모리가 성능 병목이 되는 현상

### 학습 과정에서의 메모리 사용 시점

| 단계 | 메모리 사용 |
|------|------------|
| 데이터 저장 | dataset → DRAM → GPU, batch 데이터 저장 |
| Forward | activation(중간 출력값) 저장 → backpropagation 시 필요 |
| Loss 계산 | loss 값 계산 |
| Backpropagation | 중간 layer 결과값, 각 weight에 대한 gradient |
| Weight 업데이트 | gradient → weight 반영 |

---

## 2. Methodology

### a. Melon
> *Melon: Breaking the Memory Wall for Resource-Efficient On-Device Machine Learning*

#### Lifetime-aware Memory Pool
기존 방식은 텐서를 순차적으로 할당만 하고 학습이 반복된다는 특성을 고려하지 않는다. 수명이 긴 텐서는 메모리를 오래 점유해서 다른 텐서 배치를 방해한다 (2D strip packing 문제).

- **Activation 값 (긴 수명)**: forward에서 생성, backward에서 해제 → First Produce Last Release (FPLR)
- **Temporary tensor (짧은 수명)**: 연산 중간 계산용 (add, mul, reshape 중간값)

→ 긴 lifetime 텐서를 아래쪽에, 짧은 텐서를 위쪽에 배치하여 메모리 단편화 감소

#### Recomputation
메모리 사용이 예산을 초과하면 recomputation을 트리거한다. TPS(Tensor Priority Score)가 가장 큰 tensor부터 제거한다.

```
TPS = (TensorSize × FreedLifetime) / RecomputationTime
```

- TPS 높음 → 메모리를 많이 확보하고 오래 유지되며 재계산 비용이 낮음 → 제거 1순위

#### Micro-batch
Batch Normalization 레이어가 없는 경우, 배치를 작은 단위로 나누어 순차적으로 처리함으로써 메모리 사용량을 절감

---

### b. Sage
> *Memory-efficient DNN Training on Mobile Devices*

#### Unified AD (Automatic Differentiation) Graph
forward와 backward를 하나의 그래프로 합쳐 실행 계획을 한 번에 최적화한다.

#### Operator Fusion
여러 연산을 하나의 큰 연산으로 합쳐 중간 텐서 저장을 제거한다.

```python
# fusion 없음 (기존): 최대 300~400MB
t1 = x @ W   # matmul  → 100MB
t2 = t1 + b  # add     → 100MB
t3 = t2 * 2  # multiply→ 100MB
y  = relu(t3)          → 100MB

# fusion 적용: 100MB
y = fused(x, W, b)
```

#### Dynamic Gradient Checkpointing
비트당 계산 비용(computational cost per bit) 휴리스틱 기준으로 값이 낮은 항목을 메모리에서 제거하고, 필요 시 재계산하여 사용한다.

#### Dynamic Gradient Accumulation
메모리 상황에 따라 micro-batch 크기를 자동으로 조정한다.
- 메모리 충분 → batch = 16
- 메모리 부족 → batch = 4

---

### c. Dropback
> *Full Deep Neural Network Training on a Pruned Weight Budget*

대부분의 가중치는 초기값에서 거의 변하지 않으며, 누적 gradient가 0에 가깝다는 관찰에 기반한다.

- 모든 가중치를 업데이트하지 않고 **변화가 가장 큰 상위 k%의 gradient만 추적·업데이트**
- 추적하지 않는 가중치는 freeze하여 에너지 소모가 큰 메모리 접근을 줄임
- 어떤 가중치의 gradient가 현재 추적 중인 가중치보다 커질 경우, 해당 가중치를 추적 집합에 추가하고 기존 최저 누적 gradient 가중치를 제거

---

## 3. Comparative Analysis

ResNet18 + CIFAR-10, PyTorch Profiler API를 활용하여 메모리 사용량, 학습 시간, 손실 감소, 정확도를 측정하였다.

| | Base | Melon | Sage (Checkpointing) | Sage (Accumulation) | Dropback |
|---|---|---|---|---|---|
| Memory usage (MB) | 1492 | 1506 | 866 | 568 | 562 |
| Train time (min) | 10 | 10.28 | 10.55 | 22.67 | 19.47 |
| Train loss | 1.348 | 1.353 | 1.013 | 4.804 | 1.713 |
| Validation accuracy | 53.37 | 46.17 | 68.27 | 62.47 | 37.57 |

### 분석 요약

- **Melon**: 메모리 절감 효과가 제한적이나, 장기적으로 더 큰 배치 크기를 지원할 수 있는 가능성 제공. 학습 시간 소폭 증가.
- **Sage Checkpointing**: 메모리를 866MB로 절감하면서 가장 높은 정확도(68.27%) 달성. 재연산으로 인한 학습 시간 소폭 증가.
- **Sage Accumulation**: 메모리를 568MB로 크게 절감하였으나, 학습 시간이 2배 이상 증가하고 수렴 속도 저하.
- **Dropback**: 가장 낮은 메모리(562MB)를 달성했으나, 업데이트 가중치가 적어 수렴 속도가 느리고 정확도가 낮음.

### 세 프레임워크 통합 실험 결과

세 프레임워크의 기법을 조합하여 실험한 결과:
- **메모리 사용량 462MB** 달성 → 기본 모델 대비 약 **69% 절감**
- 손실 값은 안정적으로 감소, 과적합(overfitting) 미발생
- 정확도 수렴 속도가 느려 추가적인 에포크 필요

![통합 실험 결과 (100 Epochs)](Final/Final100_results.png)

---

## 4. Conclusion

세 가지 프레임워크 모두 메모리 효율성을 향상시키는 데 기여하였으나, 학습 시간 지연 및 수렴 속도 저하라는 공통적인 한계를 보였다. 온디바이스 학습에서 **메모리 효율성과 학습 시간 간의 균형**을 맞추는 것이 핵심 과제이며, 세 프레임워크의 기법을 조합한 접근 방식이 메모리 절감에 효과적임을 확인하였다.

---

## References

- Melon: Breaking the Memory Wall for Resource-Efficient On-Device Machine Learning
- Sage: Memory-efficient DNN Training on Mobile Devices
- Dropback: Full Deep Neural Network Training on a Pruned Weight Budget
