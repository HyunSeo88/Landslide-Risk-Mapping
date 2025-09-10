# ldaps_make_daily_tifs.py
# ------------------------------------------------------------
# NetCDF 파일들을 읽어서 GeoTIFF 래스터로 변환
# ldaps_load_grib.py 출력에 맞게 수정됨
# Requires: xarray, numpy, rasterio, rioxarray
# ------------------------------------------------------------
import os, json
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import Affine
from pyproj import CRS
from collections import deque

IN_DIR = r"D:\Landslide\data\LDAPS\NetCDF"
OUT_DIR = r"D:\Landslide\data\LDAPS\ldaps_tif"
os.makedirs(os.path.join(OUT_DIR,"acc5d"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"peak1h"), exist_ok=True)

# LCC 좌표계 정의 (ldaps_load_grib.py와 동일)
LCC_CRS = CRS.from_proj4("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +datum=WGS84 +units=m +no_defs")

def estimate_transform_from_coords(ds: xr.Dataset) -> Affine:
    """좌표 정보로부터 Transform 추정"""
    try:
        # y, x 차원의 크기
        ny, nx = ds.dims['y'], ds.dims['x']
        
        # 첫 번째 데이터 변수에서 좌표 추출 시도
        data_vars = list(ds.data_vars.keys())
        if not data_vars:
            raise ValueError("No data variables found")
        
        # 단위 격자 크기 추정 (1.5km = 1500m)
        pixel_size = 1500.0  # LDAPS 격자 크기
        
        # 중심점 추정 (한국 중부 지역)
        center_x = 200000.0  # LCC 투영 좌표계에서 대략적인 중심
        center_y = 500000.0
        
        # 좌상단 모서리 계산
        left = center_x - (nx * pixel_size) / 2
        top = center_y + (ny * pixel_size) / 2
        
        return Affine.translation(left, top) * Affine.scale(pixel_size, -pixel_size)
        
    except Exception as e:
        print(f"[WARNING] Could not estimate transform: {e}")
        # 기본값 반환
        return Affine.identity()

def save_geotiff(arr: np.ndarray, out_path: str, transform: Affine, crs: CRS):
    """GeoTIFF 파일 저장"""
    # NaN 값 처리
    arr = np.where(np.isfinite(arr), arr, np.nan)
    
    h, w = arr.shape
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=h, width=w, count=1, dtype="float32",
        crs=crs, transform=transform,
        nodata=np.float32(np.nan),
        compress="deflate", predictor=2, zlevel=6, 
        tiled=True, blockxsize=512, blockysize=512
    ) as dst:
        dst.write(arr.astype(np.float32), 1)

def main():
    print("[INFO] Processing LDAPS NetCDF files to GeoTIFF...")
    
    # NetCDF 파일 목록
    files = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".nc")])
    if not files:
        raise SystemExit(f"No NetCDF files found in {IN_DIR}")
    
    print(f"[INFO] Found {len(files)} NetCDF files")

    rolling = deque(maxlen=5)  # 5일 롤링 윈도우
    daily_sums = {}            # 날짜별 일 누적 저장
    transform = None
    crs = LCC_CRS

    # 첫 파일에서 메타 정보 추출
    for i, fn in enumerate(files):
        filepath = os.path.join(IN_DIR, fn)
        
        try:
            ds = xr.open_dataset(filepath)
            
            # Transform 정보 추출 (첫 파일에서만)
            if transform is None:
                if "crs" in ds.attrs:
                    try:
                        crs = CRS.from_wkt(ds.attrs["crs"])
                    except:
                        crs = LCC_CRS
                        
                transform = estimate_transform_from_coords(ds)
                print(f"[INFO] Using CRS: {crs.to_string()}")
                print(f"[INFO] Transform: {transform}")

            # 날짜 추출 (파일명에서)
            day_str = fn.replace("LDAPS_Rain_", "").replace(".nc", "")  # YYYYMMDD 추출
            print(f"[PROCESS] {day_str} ({i+1}/{len(files)})")
            
            # 강수량 데이터 추출
            if "precipitation" in ds.data_vars:
                rain_data = ds["precipitation"].values  # shape: (24, y, x)
            elif "rain1h_mm" in ds.data_vars:
                rain_data = ds["rain1h_mm"].values
            else:
                data_vars = list(ds.data_vars.keys())
                if data_vars:
                    rain_data = ds[data_vars[0]].values
                    print(f"[INFO] Using variable: {data_vars[0]}")
                else:
                    print(f"[WARNING] No data variables found in {fn}")
                    ds.close()
                    continue
            
            # 단위 변환 (kg m**-2 s**-1 → mm/h → mm/day)
            if "kg m**-2 s**-1" in str(ds.attrs) or "precipitation" in ds.data_vars:
                # kg/m²/s를 mm/h로 변환 (1 kg/m²/s = 3600 mm/h)
                rain_data = rain_data * 3600.0
            
            # 일별 통계 계산
            # NaN 값 처리
            rain_data = np.where(np.isfinite(rain_data), rain_data, 0.0)
            
            # 일 누적 강우량 (24시간 합계)
            sum_day = np.sum(rain_data, axis=0)
            
            # 일 최대 강우강도 (시간당 최대값)
            peak_day = np.max(rain_data, axis=0)
            
            # 5일 롤링 저장
            daily_sums[day_str] = sum_day
            rolling.append(sum_day)
            
            # 5일 누적 강우량 계산
            if len(rolling) > 0:
                acc5 = np.sum(np.stack(list(rolling), axis=0), axis=0)
            else:
                acc5 = sum_day
            
            # GeoTIFF 파일로 저장
            out_acc = os.path.join(OUT_DIR, "acc5d", f"{day_str}_acc5d_mm.tif")
            out_peak = os.path.join(OUT_DIR, "peak1h", f"{day_str}_peak1h_mm.tif")
            
            save_geotiff(acc5, out_acc, transform, crs)
            save_geotiff(peak_day, out_peak, transform, crs)
            
            print(f"[SAVE] {os.path.basename(out_acc)} (5-day: {np.nanmax(acc5):.1f} mm)")
            print(f"[SAVE] {os.path.basename(out_peak)} (peak: {np.nanmax(peak_day):.1f} mm/h)")
            
            ds.close()
            
        except Exception as e:
            print(f"[ERROR] Failed to process {fn}: {e}")
            continue
    
    print(f"[COMPLETE] Processed {len(daily_sums)} days successfully!")
    print(f"[OUTPUT] 5-day accumulation: {OUT_DIR}/acc5d/")
    print(f"[OUTPUT] Daily peak intensity: {OUT_DIR}/peak1h/")

if __name__ == "__main__":
    main()
