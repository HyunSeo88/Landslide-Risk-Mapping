# Hierarchical GNN-U-Net Inference Guide

## 개요

학습된 GNN-U-Net 모델을 사용하여 픽셀 단위 산사태 위험지도를 생성합니다.

**핵심 특징:**
- 🎯 **2단계 예측**: GNN (정적 취약성) + U-Net (동적 위험도)
- 🔄 **슬라이딩 윈도우**: Overlap을 사용하여 seam artifact 방지
- 💾 **효율적**: GNN은 한 번만 실행, 동적 데이터만 날짜별로 로드
- 📊 **고해상도**: 픽셀 단위 (30m) 위험도 예측

## 사용법

### 1. 단일 날짜 예측

```bash
python src/inference/predict_unet.py \
    --checkpoint experiments/hierarchical_unet/20251016_235109/checkpoints/model_best.pth \
    --date 20200715 \
    --output outputs/risk_maps/risk_map_20200715.tif
```

### 2. 여러 날짜 일괄 예측

```bash
python src/inference/predict_unet.py \
    --checkpoint experiments/hierarchical_unet/20251016_235109/checkpoints/model_best.pth \
    --dates 20200615 20200715 20200815 \
    --output_dir outputs/risk_maps/
```

### 3. GNN 취약성 맵 함께 저장

```bash
python src/inference/predict_unet.py \
    --checkpoint experiments/hierarchical_unet/20251016_235109/checkpoints/model_best.pth \
    --date 20200715 \
    --output outputs/risk_maps/risk_map_20200715.tif \
    --save_gnn
```

출력:
- `risk_map_20200715.tif` - 최종 위험도 맵
- `risk_map_20200715_gnn_susceptibility.tif` - GNN 정적 취약성 맵

### 4. CPU에서 실행

```bash
python src/inference/predict_unet.py \
    --checkpoint experiments/hierarchical_unet/20251016_235109/checkpoints/model_best.pth \
    --date 20200715 \
    --output outputs/risk_maps/risk_map_20200715.tif \
    --device cpu
```

### 5. 패치 크기 및 Overlap 조정

```bash
python src/inference/predict_unet.py \
    --checkpoint experiments/hierarchical_unet/20251016_235109/checkpoints/model_best.pth \
    --date 20200715 \
    --output outputs/risk_maps/risk_map_20200715.tif \
    --patch_size 1024 \
    --overlap 128
```

## 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--checkpoint` | 학습된 모델 체크포인트 경로 (.pth) | 필수 |
| `--date` | 단일 날짜 (YYYYMMDD) | - |
| `--dates` | 여러 날짜 (YYYYMMDD YYYYMMDD ...) | - |
| `--output` | 단일 날짜 출력 경로 (.tif) | - |
| `--output_dir` | 여러 날짜 출력 디렉토리 | `outputs/risk_maps` |
| `--device` | 사용 디바이스 (cuda/cpu) | `cuda` |
| `--patch_size` | 슬라이딩 윈도우 패치 크기 | `512` |
| `--overlap` | 패치 간 겹침 크기 | `64` |
| `--save_gnn` | GNN 취약성 맵 별도 저장 | `False` |

## 슬라이딩 윈도우 동작 원리

```
전체 래스터: 4862 × 5040 픽셀

┌─────────────────────────────────────┐
│  ┌───────┐                           │
│  │ Patch │                           │
│  │  1    │  ← 512×512 패치           │
│  └───────┘                           │
│      ┌───────┐                       │
│      │ Patch │  ← 64 픽셀 겹침       │
│      │  2    │                       │
│      └───────┘                       │
│          ┌───────┐                   │
│          │ Patch │                   │
│          │  3    │                   │
│          └───────┘                   │
│              ...                     │
└─────────────────────────────────────┘

Stride = 512 - 64 = 448 픽셀
→ 겹치는 부분은 평균하여 부드러운 결과
```

## 출력 형식

### Risk Map (GeoTIFF)

- **Format**: GeoTIFF (Float32)
- **CRS**: EPSG:5179 (Korea 2000 / Central Belt)
- **Resolution**: 30m × 30m
- **Values**: 0.0 ~ 1.0 (위험도 확률)
- **Compression**: LZW

### 값 해석

| 값 범위 | 위험도 | 색상 (예시) |
|---------|--------|-------------|
| 0.0 - 0.2 | 매우 낮음 | 녹색 |
| 0.2 - 0.4 | 낮음 | 연두 |
| 0.4 - 0.6 | 보통 | 노란색 |
| 0.6 - 0.8 | 높음 | 주황색 |
| 0.8 - 1.0 | 매우 높음 | 빨간색 |

## 처리 시간 예상

**RTX 4070 Ti (12GB) 기준:**

| 설정 | 전체 영역 (4862×5040) | 시간 |
|------|----------------------|------|
| 패치 512, Overlap 64 | ~120 패치 | ~2분 |
| 패치 1024, Overlap 128 | ~30 패치 | ~1분 |
| CPU (32코어) | ~120 패치 | ~15분 |

**GNN Forward는 한 번만 실행** (~5초) → 여러 날짜 예측 시 효율적!

## 예제 워크플로우

### 시나리오: 2020년 여름 기간 위험도 분석

```bash
# 1. 학습된 모델로 6-9월 위험도 맵 생성
python src/inference/predict_unet.py \
    --checkpoint experiments/hierarchical_unet/best_model/checkpoints/model_best.pth \
    --dates 20200601 20200615 20200701 20200715 20200801 20200815 20200901 \
    --output_dir outputs/risk_maps/summer_2020/ \
    --save_gnn

# 2. QGIS에서 시각화
# - outputs/risk_maps/summer_2020/*.tif 로드
# - 팔레트: YlOrRd (노란색-주황-빨강)
# - Min: 0, Max: 1
# - 반투명 overlay로 지형도 위에 표시

# 3. 시계열 분석
# - 각 날짜별 고위험 픽셀 (>0.8) 개수 추출
# - 강우량 데이터와 비교
```

## 문제 해결

### CUDA Out of Memory

```bash
# 패치 크기 줄이기
--patch_size 256 --overlap 32

# 또는 CPU 사용
--device cpu
```

### 동적 래스터 파일 없음

```
Warning: {file} not found, using zeros
```

→ 해당 날짜의 동적 데이터(강우 등)가 없으면 0으로 채움  
→ GNN 취약성만으로 예측 (정적 요인만 고려)

### 메모리 부족 (RAM)

큰 래스터 로드 시 RAM 부족:
```bash
# 패치 크기를 더 작게
--patch_size 128 --overlap 16
```

## 추가 기능

### Python API 사용

```python
from src.inference.predict_unet import UNetPredictor

# 예측기 초기화
predictor = UNetPredictor(
    checkpoint_path='experiments/.../model_best.pth',
    device='cuda',
    patch_size=512,
    overlap=64
)

# 예측
risk_map = predictor.predict(
    date='20200715',
    output_path='outputs/risk_map.tif',
    save_gnn_susceptibility=True
)

# NumPy 배열로 반환됨
print(risk_map.shape)  # (4862, 5040)
print(risk_map.min(), risk_map.max())
```

### GNN 취약성 맵 재사용

```python
# GNN은 한 번만 계산되고 캐싱됨
predictor.predict(date='20200615', output_path='out1.tif')  # GNN 계산
predictor.predict(date='20200715', output_path='out2.tif')  # GNN 재사용 (빠름!)
predictor.predict(date='20200815', output_path='out3.tif')  # GNN 재사용
```

## 참고

- **학습 코드**: `src/models/train_unet.py`
- **모델 구조**: `src/models/model_unet.py`
- **데이터 로더**: `src/models/data_loader_mil.py`
- **Config 예시**: `configs/hierarchical_unet.yaml`

---
**Last Updated**: 2025-01-16  
**Author**: Landslide Risk Analysis Project Team

