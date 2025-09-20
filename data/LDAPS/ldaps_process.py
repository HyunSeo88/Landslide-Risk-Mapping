# ldaps_process.py
# ------------------------------------------------------------
# 개선된 NetCDF 파일들을 rioxarray를 사용해서 WGS84 GeoTIFF로 변환
# --- 개선사항 ---
# 1) CF-Conventions grid_mapping 변수 인식
# 2) 압축/청킹된 NetCDF 효율적 처리
# 3) 정확한 좌표계 자동 인식
# 4) 5일 롤링 및 피크 계산 추가
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from collections import deque
import warnings

# rioxarray 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='rioxarray')

# 입출력 디렉토리
IN_DIR = r"D:\Landslide\data\LDAPS\NetCDF"
OUT_DIR = r"D:\Landslide\data\LDAPS\ldaps_tif_wgs84"

# 출력 디렉토리 생성
os.makedirs(os.path.join(OUT_DIR, "acc5d"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "peak1h"), exist_ok=True)

def detect_crs_from_netcdf(ds):
    """NetCDF에서 CRS 정보 자동 감지"""
    
    # 1) CF-Convention grid_mapping 변수 확인
    if 'crs_wgs84' in ds.data_vars:
        crs_var = ds['crs_wgs84']
        if hasattr(crs_var, 'attrs') and 'grid_mapping_name' in crs_var.attrs:
            grid_mapping_name = crs_var.attrs.get('grid_mapping_name')
            if grid_mapping_name == 'latitude_longitude':
                print("[INFO] Detected WGS84 from CF grid_mapping")
                return "EPSG:4326"
    
    # 2) 좌표 이름으로 추정
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        print("[INFO] Detected lat/lon coordinates, assuming WGS84")
        return "EPSG:4326"
    
    # 3) 기본값
    print("[WARNING] Could not detect CRS, defaulting to WGS84")
    return "EPSG:4326"

def setup_spatial_coordinates(data_array, ds):
    """공간 좌표 설정 - 2D lat/lon을 rioxarray가 인식할 수 있도록 처리"""
    
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lats_2d = ds.latitude.values
        lons_2d = ds.longitude.values
        
        print(f"[INFO] Grid shape: {lats_2d.shape}")
        print(f"[INFO] Lat: {lats_2d.min():.3f}~{lats_2d.max():.3f}")
        print(f"[INFO] Lon: {lons_2d.min():.3f}~{lons_2d.max():.3f}")
        
        # rioxarray를 위한 좌표 설정
        # 2D lat/lon이 있으면 y, x 차원으로 평균값 사용
        if lats_2d.ndim == 2 and lons_2d.ndim == 2:
            # 각 y, x 인덱스에 해당하는 대표 lat/lon 계산
            lats_1d = lats_2d.mean(axis=1)  # 각 y에 대한 평균 lat
            lons_1d = lons_2d.mean(axis=0)  # 각 x에 대한 평균 lon
            
            # 좌표 재할당
            data_array = data_array.assign_coords(y=lats_1d, x=lons_1d)
            print(f"[INFO] Assigned 1D coordinates from 2D lat/lon")
    
    return data_array

def main():
    """메인 처리 함수"""
    print(f"[INFO] Starting improved LDAPS NetCDF to WGS84 GeoTIFF conversion...")
    print(f"[INFO] Input: {IN_DIR}")
    print(f"[INFO] Output: {OUT_DIR}")
    
    # NetCDF 파일 목록
    files = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".nc")])
    if not files:
        raise SystemExit(f"No NetCDF files found in {IN_DIR}")
    
    print(f"[INFO] Found {len(files)} NetCDF files")
    
    rolling = deque(maxlen=5)  # 5일 롤링 윈도우
    daily_sums = {}

    # 각 파일 처리
    for i, filename in enumerate(files):
        filepath = os.path.join(IN_DIR, filename)
        
        try:
            # 날짜 추출
            day_str = filename.replace("LDAPS_Rain_", "").replace(".nc", "")
            print(f"\n[PROCESS] {day_str} ({i+1}/{len(files)})")
            
            # NetCDF 파일 열기
            with xr.open_dataset(filepath) as ds:
                
                print(f"[INFO] Dataset info:")
                print(f"  - Variables: {list(ds.data_vars.keys())}")
                print(f"  - Coordinates: {list(ds.coords.keys())}")
                print(f"  - Attributes: CF={ds.attrs.get('Conventions', 'None')}")
                
                # CRS 자동 감지
                detected_crs = detect_crs_from_netcdf(ds)
                
                # 강수량 변수 찾기
                precip_vars = [v for v in ds.data_vars.keys() if 'precip' in v.lower()]
                if not precip_vars:
                    precip_vars = list(ds.data_vars.keys())
                
                if not precip_vars:
                    print(f"[WARNING] No data variables found in {filename}")
                    continue
                
                var_name = precip_vars[0]
                print(f"[INFO] Using variable: {var_name}")
                
                # 데이터 추출
                rain_data = ds[var_name]
                
                # 공간 좌표 설정
                rain_data = setup_spatial_coordinates(rain_data, ds)
                
                # CRS 설정
                rain_data = rain_data.rio.write_crs(detected_crs)
                
                # 단위 변환 (kg m-2 s-1 → mm/h)
                if rain_data.attrs.get('units') == 'kg m-2 s-1':
                    rain_data = rain_data * 3600.0
                    rain_data.attrs['units'] = 'mm/h'
                    print(f"[INFO] Converted units: kg m-2 s-1 → mm/h")
                
                # 시간 차원 처리
                if "time" in rain_data.dims:
                    # 24시간 통계
                    sum_day = rain_data.sum(dim="time", skipna=True)
                    sum_day.attrs['long_name'] = 'Daily accumulated precipitation'
                    sum_day.attrs['units'] = 'mm'
                    
                    peak_day = rain_data.max(dim="time", skipna=True) 
                    peak_day.attrs['long_name'] = 'Daily peak precipitation rate'
                    peak_day.attrs['units'] = 'mm/h'
                    
                    print(f"[INFO] Computed daily stats from {rain_data.time.size} time steps")
                else:
                    sum_day = rain_data
                    peak_day = rain_data
                
                # 5일 롤링
                daily_sums[day_str] = sum_day
                rolling.append(sum_day)
                
                # 5일 누적
                if len(rolling) > 1:
                    acc5 = xr.concat(list(rolling), dim="rolling_day").sum(dim="rolling_day", skipna=True)
                    acc5.attrs['long_name'] = '5-day accumulated precipitation'
                    acc5.attrs['units'] = 'mm'
                else:
                    acc5 = sum_day
                
                # GeoTIFF 저장
                out_acc = os.path.join(OUT_DIR, "acc5d", f"{day_str}_acc5d_mm_wgs84.tif")
                out_peak = os.path.join(OUT_DIR, "peak1h", f"{day_str}_peak1h_mm_wgs84.tif")
                
                # 고품질 압축 옵션
                raster_options = {
                    'compress': 'deflate',
                    'predictor': 2,
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256,
                    'BIGTIFF': 'IF_NEEDED'
                }
                
                acc5.rio.to_raster(out_acc, **raster_options)
                peak_day.rio.to_raster(out_peak, **raster_options)
                
                # 통계 출력
                acc5_max = float(acc5.max().values) if not np.isnan(acc5.max().values) else 0.0
                peak_max = float(peak_day.max().values) if not np.isnan(peak_day.max().values) else 0.0
                
                print(f"[SAVE] {os.path.basename(out_acc)} (max: {acc5_max:.1f} mm)")
                print(f"[SAVE] {os.path.basename(out_peak)} (max: {peak_max:.1f} mm/h)")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[COMPLETE] Processed {len(daily_sums)} days successfully!")
    print(f"[OUTPUT] Files saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()