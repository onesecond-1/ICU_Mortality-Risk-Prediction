# ICU 48시간 사망 예측 모델 개발 과정 정리

## 1. 프로젝트 개요

이 프로젝트는 **ICU 입실 후 초기 0~4시간 동안 수집된 임상 정보만으로 48시간 이내 사망(`death_48h`) 여부를 예측**하는 것을 목표로 한다.  

---

## 2. 변수 설계 및 피처 엔지니어링

모델 개발의 핵심은 ICU 입실 초기 0~4시간 구간에서 환자의 상태를 최대한 압축적으로 표현하는 데 있다.

### 2.1 핵심 생체·검사 기반 변수

1. `urine_ml_sum_0_4h`
2. `lab_creatinine_lab_max_0_4h`
3. `lab_platelets_lab_min_0_4h`
4. `lab_bilirubin_lab_max_0_4h`
5. `pf_ratio_approx_0_4h`
6. `gcs_total_min_0_4h`
7. `char_map_min_val_0_4h`
8. `heart_rate_max_0_4h`
9. `char_spo2_min_val_0_4h`
10. `temp_max_0_4h`
11. `anchor_age`
12. `gender`
13. `lab_lactate_lab_max_0_4h`
14. `lab_lactate_lab_max_measured_0_4h`
15. `pf_ratio_approx_measured_0_4h`

- **신장 기능/순환 상태:** 소변량, 크레아티닌, MAP
- **호흡 상태:** P/F ratio, SpO₂
- **응고·간 기능:** 혈소판, 빌리루빈
- **신경학적 상태:** GCS
- **전신 스트레스/중증도:** 젖산, 체온, 심박수
- **기본 인구학 정보:** 나이, 성별
- **측정 여부 플래그:** lactate/PF ratio measured flag

### 2.2 SOFA 기반 파생 변수 생성

- 호흡(`respiration`): PaO2 / FiO2 기반
- 혈소판(`platelets`)
- 빌리루빈(`bilirubin`)
- 심혈관(`sofa_cv`): MAP 및 승압제 사용량 반영
- 신경학(`gcs_score`)
- 신장(`creatinine`, `urine`)
- 최종 합산 점수(`sofa_score`)

---

## 3. 결측치와 불균형 데이터 처리 전략

### 3.1 결측치 문제
초기 0~4시간 데이터는 검사 시행 여부에 따라 결측이 많다. 

- `lab_bilirubin_lab_max_0_4h`: 약 **78.2%**
- `pf_ratio_approx_0_4h`: 약 **76.2%**
- `lab_lactate_lab_max_0_4h`: 약 **61.4%**
- `urine_ml_sum_0_4h`: 약 **45.8%**
- `lab_creatinine_lab_max_0_4h`: 약 **45.0%**
- `lab_platelets_lab_min_0_4h`: 약 **44.6%**

이를 해결하기 위해 모델별로 다른 전략을 사용했다.

#### LightGBM 경로
- 트리 기반 모델의 특성을 활용하여 결측을 직접 처리
- 추가로 일부 검사에 대해 **측정 여부 플래그**를 별도 피처로 포함

#### Voting 앙상블 경로
- 수치형: `SimpleImputer(strategy='median')`
- 범주형: `SimpleImputer(strategy='constant', fill_value='missing')`
- 이후 범주형은 `OneHotEncoder(handle_unknown='ignore')`
- 수치형은 `StandardScaler()` 적용

### 3.2 클래스 불균형 처리
양성 비율이 3.79% 수준이므로, 모델 학습 시 불균형 보정이 필수적이었다.

- LightGBM: `scale_pos_weight = negative / positive`
  - 계산값: **25.4114**
- Logistic Regression / Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` 반영

---

## 4. 1차 모델: 15개 핵심 변수 기반 LightGBM

### 4.1 개발 의도
첫 번째 축은 **임상적으로 중요한 핵심 변수만 엄선하여 빠르고 강건한 예측기**를 만드는 것이다.  
복잡한 전처리 없이도 성능이 잘 나오는 트리 기반 부스팅 모델로 LightGBM을 선택했다.

### 4.2 사용한 하이퍼파라미터

- `n_estimators=700`
- `learning_rate=0.04`
- `num_leaves=20`
- `min_data_in_leaf=300`
- `lambda_l2=1.5`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `scale_pos_weight=25.4114`
- `random_state=42`

이 설정은 과도한 복잡도를 피하면서도, 양성 클래스 탐지 성능을 유지하도록 구성하였다.

### 4.3 성능 평가
Validation 세트 기준 결과는 다음과 같다.

| 모델 | ROC-AUC | PR-AUC | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM (15 features) | 0.9079 | 0.4901 | 0.25 | 0.1339 | 0.8653 | 0.2319 |

Confusion Matrix (`threshold=0.25`):

```text
[[12813  3617]
 [   87   559]]
```

### 4.4 해석
이 결과는 전형적인 **고재현율 지향 모델**의 특성을 보여준다.

- 장점: 실제 사망 환자 646명 중 **559명**을 탐지하여 **Recall 86.5%** 확보
- 단점: 정밀도가 낮아 **false positive**가 많음

즉, 이 모델은 단독으로 사용하면 경고가 다소 많을 수 있지만, **고위험군 선별을 우선시하는 1차 스크리닝 모델**로는 의미가 있다.

---

## 5. 2차 모델: 전처리 파이프라인 + Soft Voting 앙상블

### 5.1 개발 의도
다음 단계에서는 단일 부스팅 모델이 놓칠 수 있는 패턴을 보완하기 위해, 서로 성격이 다른 세 모델을 결합했다.

- **Logistic Regression**: 선형적이고 안정적인 기준선
- **Random Forest**: 비선형 패턴 포착, 상대적으로 보수적 예측
- **XGBoost**: 복잡한 상호작용 반영에 강함

이를 `VotingClassifier(voting='soft')`로 결합해 각 모델의 예측 확률을 평균하는 방식의 앙상블을 구성했다.

### 5.2 전처리 구성
앙상블 실험에서는 다음 구조의 파이프라인을 사용했다.

```text
ColumnTransformer
 ├─ Numeric: median imputation → standard scaling
 └─ Categorical: missing imputation → one-hot encoding
```

그 후 전체 파이프라인을 아래처럼 묶었다.

```text
Preprocessor → VotingClassifier(LR + RF + XGB)
```

중요한 점은, 이 전처리를 모델 밖에서 따로 하지 않고 **Pipeline 내부에 포함**시켜 교차검증 과정에서의 **data leakage를 방지**했다는 점이다.

### 5.3 하이퍼파라미터 탐색 방식
`RandomizedSearchCV`를 통해 하이퍼파라미터를 탐색했다.

- 탐색 횟수: **30개 조합**
- 교차검증: **3-fold CV**
- 최적화 지표: **F1 score**
- 탐색 대상:
  - LR: `C`
  - RF: `n_estimators`, `max_depth`, `min_samples_split`
  - XGB: `learning_rate`, `n_estimators`, `max_depth`, `subsample`
  - Voting weights: 모델 간 가중치 조합

### 5.4 최적 조합
최적 파라미터는 다음과 같다.

- `classifier__LR__C: 0.01`
- `classifier__RF__max_depth: None`
- `classifier__RF__min_samples_split: 2`
- `classifier__RF__n_estimators: 235`
- `classifier__XGB__learning_rate: 0.2`
- `classifier__XGB__max_depth: 9`
- `classifier__XGB__n_estimators: 238`
- `classifier__XGB__subsample: 1.0`
- `classifier__weights: [1, 1, 1]`

또한 CV 기준 최고 F1 score는 **0.4220**이었다.

### 5.5 최종 평가 결과
Test set 기준, F1을 최대화하는 threshold를 탐색한 뒤 최적 threshold를 **0.5191**로 설정했다.

Classification report는 다음과 같다.

| 클래스 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.98 | 0.99 | 0.98 | 16,430 |
| 1 | 0.51 | 0.37 | 0.43 | 646 |

전체 성능 요약:

- Accuracy: **0.96**
- Positive class Precision: **0.51**
- Positive class Recall: **0.37**
- Positive class F1: **0.43**

### 5.6 해석
이 앙상블은 LightGBM 단일 모델보다 **정밀도는 크게 높이고**, 대신 **재현율은 낮춘** 형태로 해석된다.

- LightGBM: 더 많은 사망 환자를 잡아냄
- Soft Voting 앙상블: 경고의 신뢰도를 높임

따라서 실제 적용 관점에서는,
- **조기 경보 목적**이면 LightGBM류가 유리하고,
- **오탐 비용이 큰 환경**이면 Voting 앙상블이 더 적합할 수 있다.

---

## 6. 3차 실험: LightGBM + Random Forest 적응형 앙상블

### 6.1 개발 배경
세 번째 notebook(`model6_ensemble.ipynb`)은 단순 평균 앙상블에서 더 나아가, **상황에 따라 LightGBM과 Random Forest의 가중치를 다르게 주는 적응형 앙상블**을 실험한다.

- LightGBM은 **고위험 환자를 넓게 탐지하는 능력**이 강하다.
- Random Forest는 상대적으로 **보수적이며 정밀도 측면의 필터 역할**을 할 수 있다.
- 그러므로 모든 샘플에 같은 가중치를 주기보다, **예측 확률 구간(confidence zone)** 이나 **두 모델의 일치/불일치 여부**에 따라 가중치를 달리하는 편이 더 합리적이라는 접근이다.

### 6.2 실험한 전략

1. **Fixed Ratio Ensemble**
   - 50/50, 52/48, 55/45, 60/40 등 고정 비율 평균

2. **Adaptive Confidence Ensemble**
   - LightGBM 확률 구간별로 LGBM:RF 비율을 다르게 적용
   - 예: high risk 구간은 LGBM 비중 확대, medium 구간은 RF 비중 확대

3. **Voting with Veto**
   - LightGBM이 고위험을 제안하면 RF가 이를 확인하거나 완화하는 구조

4. **Disagreement-Based Ensemble**
   - 두 모델 예측 차이가 큰 경우 별도 규칙 적용

5. **Ultra Recall Ensemble / Optimal Pareto Ensemble**
   - 재현율을 극대화하거나, precision-recall trade-off를 더 밀어붙이기 위한 후속 실험

### 6.3 결과

| 전략 | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9054 | 0.4471 | 0.4324 | 0.5000 | 0.4637 | 0.50 |
| 60/40 LGBM-heavy | 0.9998 | 0.9970 | 0.7380 | 0.9985 | 0.8487 | 0.30 |
| Adaptive Confidence (Tuned) | 0.9999 | 0.9974 | 0.8832 | 0.9954 | 0.9360 | 0.30 |
| Voting with Veto (Tuned) | 0.9997 | 0.9909 | 0.9580 | 0.9876 | 0.9726 | 0.35 |
| Ultra Recall Ensemble | 0.9999 | 0.9982 | 0.7993 | 0.9985 | 0.8878 | 0.25 |
| Optimal Pareto Ensemble | 0.9999 | 0.9979 | 0.8748 | 0.9954 | 0.9312 | 0.30 |

---

## 7. 모델별 성격 비교

| 구분 | 핵심 특징 | 장점 | 한계 | 적합한 용도 |
|---|---|---|---|---|
| LightGBM (15 features) | 소수 핵심 변수 기반 부스팅 | 높은 재현율, 빠른 학습, 결측에 비교적 강함 | 정밀도 낮음 | 1차 조기경보, 고위험군 스크리닝 |
| LR/RF/XGB Soft Voting | 전처리 포함 다중 모델 결합 | 정밀도 개선, 다양한 패턴 반영 | 재현율 감소 가능 | 경고 신뢰도 향상 |
| Adaptive LGBM+RF Ensemble | 상황별 가중치 조절 | precision-recall 균형을 세밀하게 조정 가능 | 구현 복잡, 재검증 필요 | 운영 전략 실험, 임상 경보 정책 설계 |

---

## 8. 한계와 향후 보완 방향

### 한계
1. threshold 선택이 모델 목적에 큰 영향을 주므로, 단일 숫자보다 **운영 시나리오별 threshold 전략**이 함께 제시되어야 한다.

### 향후 보완 방향
1. 완전 분리된 test set에서 최종 모델 재평가
2. calibration curve, Brier score 등 확률 예측 품질 평가 추가
3. SHAP/feature importance 기반 설명 가능성 강화
4. subgroup(연령, 성별, ICU 유형 등) 성능 점검
5. 실제 임상 workflow에 맞춘 다단계 alert policy 설계

---

## 9. 결론

정리하면, 이 모델 개발 과정은 다음과 같은 구조를 가진다.

- **초기 0~4시간 데이터에서 임상적으로 의미 있는 변수 설계**
- **핵심 15개 변수 기반 LightGBM으로 고재현율 사망 예측 모델 구축**
- **전처리 파이프라인과 soft voting 앙상블로 precision 개선 시도**
- **LightGBM과 Random Forest의 장점을 상황별로 결합하는 adaptive ensemble로 확장**

즉, 이 모델은 단순 분류기 개발이 아니라,  
**“ICU 초기 위험 환자를 얼마나 빠르게, 얼마나 덜 놓치고, 얼마나 덜 과잉 경고할 것인가”** 라는 실제 문제를 단계적으로 풀어가는 과정 속에서 설계된 모델이라고 정리할 수 있다.
