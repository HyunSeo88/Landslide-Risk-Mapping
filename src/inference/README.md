# Inference & Analysis Scripts

학습된 산사태 위험도 모델을 사용하여 예측 및 분석을 수행하는 스크립트 모음입니다.

## 📁 파일 구조

```
src/inference/
├── predict.py              # 예측 및 평가 스크립트
├── run_shap_analysis.py    # SHAP 변수 중요도 분석
└── README.md              # 이 파일
```

## 🚀 사용 방법

### 1. 모델 예측 및 평가

학습된 모델로 예측을 수행하고 평가 리포트를 생성합니다.

#### 기본 사용법 (학습 모드)

```bash
python src/inference/predict.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs
```

#### 실시간 예측 모드 (당일 데이터 제외)

```bash
python src/inference/predict.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs \
    --realtime_mode
```

#### 옵션 설명

- `--checkpoint`: 학습된 모델 체크포인트 경로 (필수)
- `--output_dir`: 결과 저장 디렉토리 (기본값: `outputs`)
- `--device`: 사용할 디바이스 (기본값: `cuda`)
- `--batch_size`: 배치 크기 (기본값: 256)
- `--realtime_mode`: 실시간 예측 모드 활성화 (당일 데이터 제외)

#### 생성되는 결과물

```
outputs/
├── predictions/
│   ├── predictions_training.csv      # 예측 결과 (학습 모드)
│   └── predictions_realtime.csv      # 예측 결과 (실시간 모드)
└── reports/
    ├── evaluation_report_training.txt  # 평가 리포트
    ├── roc_curve_training.png          # ROC 곡선
    └── risk_distribution_training.png  # 위험도 분포
```

**predictions CSV 포함 정보:**
- `cat`: 사면 단위 ID
- `event_date`: 이벤트 날짜
- `true_label`: 실제 라벨 (0: 안정, 1: 산사태)
- `predicted_prob`: 예측 확률 (0~1)
- `predicted_label`: 예측 라벨 (0 or 1)
- `attention_gnn`: GNN 어텐션 가중치
- `attention_rnn`: RNN 어텐션 가중치

---

### 2. SHAP 변수 중요도 분석

SHAP (SHapley Additive exPlanations)를 사용하여 모델의 변수 중요도를 분석합니다.

#### 기본 사용법

```bash
python src/inference/run_shap_analysis.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs/shap_analysis
```

#### 샘플 수 조정

```bash
python src/inference/run_shap_analysis.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs/shap_analysis \
    --n_test 200 \
    --n_background 200
```

#### 옵션 설명

- `--checkpoint`: 학습된 모델 체크포인트 경로 (필수)
- `--output_dir`: 결과 저장 디렉토리 (기본값: `outputs/shap_analysis`)
- `--device`: 사용할 디바이스 (기본값: `cuda`)
- `--n_test`: SHAP 분석할 테스트 샘플 수 (기본값: 100)
- `--n_background`: SHAP 배경 샘플 수 (기본값: 100)
- `--seed`: 랜덤 시드 (기본값: 42)

#### 생성되는 결과물

```
outputs/shap_analysis/
├── shap_dynamic_summary.png           # 동적 변수 SHAP 요약
├── shap_dynamic_importance.png        # 동적 변수 중요도 (시간 통합)
├── shap_temporal_importance.png       # 시간별 중요도 패턴
├── shap_static_summary.png            # 정적 변수 SHAP 요약
├── shap_static_importance.png         # 정적 변수 중요도
├── shap_integrated_comparison.png     # 통합 비교
└── shap_report.txt                    # 상세 텍스트 리포트
```

#### SHAP 분석 구조

**Stage 1: 동적 변수 (Dynamic Features)**
- 강우 관련 시계열 특성 (5일 window)
- 시간적 패턴 분석
- 각 날짜별 중요도

**Stage 2: 정적 변수 (Static Features)**
- 지형, 지질, 토지피복 등
- Proxy 모델을 통한 분석
- 공간적 취약성 요인

**Stage 3: 통합 분석**
- 동적 vs 정적 변수 비교
- 전체 기여도 분석
- 상세 리포트 생성

---

## 📊 결과 해석

### 예측 결과

1. **AUC-ROC**: 모델의 전반적인 분류 성능
   - 0.9 이상: 매우 우수
   - 0.8~0.9: 우수
   - 0.7~0.8: 양호

2. **Precision vs Recall**:
   - High Precision: 예측한 산사태 중 실제 산사태 비율
   - High Recall: 실제 산사태 중 예측한 비율

3. **Attention Weights**:
   - `attention_gnn`: 공간적 정보의 기여도
   - `attention_rnn`: 시간적 정보의 기여도

### SHAP 분석 결과

1. **동적 변수 중요도**:
   - `acc7d_mean/max`: 7일 누적 강우량 (장기 누적)
   - `acc3d_mean/max`: 3일 누적 강우량 (단기 누적)
   - `peak1h_mean/max`: 최대 시간당 강우량 (강우 강도)

2. **정적 변수 중요도**:
   - `slope_average`: 경사도
   - `dem_average`: 표고
   - `twi_average`: 지형습윤지수
   - `dist_stream`: 하천으로부터의 거리

3. **시간 패턴**:
   - Day -4 ~ Day 0: 어느 시점의 강우가 가장 중요한지
   - 일반적으로 당일(Day 0)과 전날(Day -1)이 중요

---

## 🔧 고급 사용법

### 특정 기간 예측

```python
# 예측 스크립트 수정 예시
# config의 start_date, end_date를 변경하여 특정 기간만 예측 가능
```

### 배치 예측

여러 체크포인트를 순회하며 예측:

```bash
for checkpoint in experiments/baseline_sage/*/checkpoints/model_best.pth; do
    echo "Processing: $checkpoint"
    python src/inference/predict.py \
        --checkpoint "$checkpoint" \
        --output_dir "outputs/$(basename $(dirname $(dirname $checkpoint)))"
done
```

---

## ⚠️ 주의사항

1. **메모리 사용량**:
   - SHAP 분석은 메모리를 많이 사용합니다
   - `--n_test`, `--n_background` 값을 조정하여 메모리 사용량 조절

2. **실시간 모드 vs 학습 모드**:
   - **학습 모드**: 당일 데이터 포함 (과거 분석용)
   - **실시간 모드**: 당일 데이터 제외 (실제 예보용)

3. **SHAP 설치**:
   ```bash
   pip install shap
   ```

---

## 📝 예제 워크플로우

### 전체 분석 파이프라인

```bash
# 1. 모델 예측 (학습 모드)
python src/inference/predict.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs/analysis_20251016

# 2. 모델 예측 (실시간 모드)
python src/inference/predict.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs/analysis_20251016 \
    --realtime_mode

# 3. SHAP 분석
python src/inference/run_shap_analysis.py \
    --checkpoint experiments/baseline_sage/20251016_005517/checkpoints/model_best.pth \
    --output_dir outputs/analysis_20251016/shap \
    --n_test 200 \
    --n_background 200
```

---

## 📚 참고 자료

- SHAP 공식 문서: https://shap.readthedocs.io/
- 모델 아키텍처: `docs/model_architecture/GNN_RNN_Hybrid_Architecture.md`
- 학습 설정: `configs/baseline.yaml`

