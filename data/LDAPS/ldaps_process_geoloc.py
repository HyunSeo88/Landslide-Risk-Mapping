# ldaps_process_geoloc.py
# ------------------------------------------------------------
# LDAPS 커브리니어 격자 → WGS84 GeoTIFF 정확한 지오로케이션 워핑
# --- 핵심 수정사항 ---
# 1) 커브리니어 격자를 그대로 보존하여 GDAL geoloc 워핑 적용
# 2) assign 금지, 실제 좌표변환(reproject/warp) 수행
# 3) precipitation 변수 명시 선택, 단위/메타데이터 승계
# 4) 시간 집계 품질 관리 추가
# ------------------------------------------------------------
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import contextlib
from collections import deque
import warnings

# GDAL 설정은 GDAL import 이후에 수행 (아래 PATH/ENV 세팅 먼저)
# 로그 파일 경로는 환경변수로도 전달
os.environ.setdefault('CPL_LOG', os.path.join(tempfile.gettempdir(), 'gdal.log'))

# Conda 환경 비활성 실행 시 GDAL 플러그인/데이터 경로 수동 설정
conda_prefix = sys.prefix
# 우선순위: Library/bin/gdalplugins → Library/lib/gdalplugins
gdal_plugins_dir_bin = os.path.join(conda_prefix, 'Library', 'bin', 'gdalplugins')
gdal_plugins_dir_lib = os.path.join(conda_prefix, 'Library', 'lib', 'gdalplugins')
gdal_data_dir = os.path.join(conda_prefix, 'Library', 'share', 'gdal')
proj_data_dir = os.path.join(conda_prefix, 'Library', 'share', 'proj')

gdal_plugins_dir = None
if os.path.isdir(gdal_plugins_dir_bin):
    gdal_plugins_dir = gdal_plugins_dir_bin
elif os.path.isdir(gdal_plugins_dir_lib):
    gdal_plugins_dir = gdal_plugins_dir_lib

if gdal_plugins_dir:
    os.environ.setdefault('GDAL_DRIVER_PATH', gdal_plugins_dir)
    print(f"[DEBUG] GDAL_DRIVER_PATH set to: {gdal_plugins_dir}")
else:
    print(f"[WARNING] GDAL plugins directory not found under {conda_prefix}")

if os.path.isdir(gdal_data_dir):
    os.environ.setdefault('GDAL_DATA', gdal_data_dir)
    print(f"[DEBUG] GDAL_DATA set to: {gdal_data_dir}")

if os.path.isdir(proj_data_dir):
    os.environ.setdefault('PROJ_LIB', proj_data_dir)
    print(f"[DEBUG] PROJ_LIB set to: {proj_data_dir}")

# DLL 의존성 로딩을 위해 PATH 보강 (conda env의 Library/bin 등)
bin_paths = [
    os.path.join(conda_prefix, 'Library', 'bin'),
    os.path.join(conda_prefix, 'Library', 'usr', 'bin'),
    os.path.join(conda_prefix, 'Library'),
    os.path.join(conda_prefix, 'Scripts'),
    os.path.join(conda_prefix, 'bin'),
]
current_path = os.environ.get('PATH', '')
augmented = os.pathsep.join(p for p in bin_paths if os.path.isdir(p) and p not in current_path)
if augmented:
    os.environ['PATH'] = augmented + os.pathsep + current_path
    print(f"[DEBUG] PATH augmented with: {augmented}")

# Windows 전용: DLL 검색 경로 우선순위 지정
if hasattr(os, 'add_dll_directory'):
    bin_dir = os.path.join(conda_prefix, 'Library', 'bin')
    if os.path.isdir(bin_dir):
        try:
            os.add_dll_directory(bin_dir)
            print(f"[DEBUG] add_dll_directory: {bin_dir}")
        except Exception:
            pass

# 문제되는 플러그인 회피 (버전 불일치 경고 방지)
os.environ.setdefault('GDAL_SKIP', 'NUMPY')
# pyproj/PROJ 경로 인식 강화
if os.path.isdir(proj_data_dir):
    os.environ.setdefault('PROJ_DATA', proj_data_dir)
os.environ.setdefault('PYPROJ_GLOBAL_CONTEXT', 'ON')

# 이제 GDAL을 import (위 PATH/ENV가 반영된 상태)
from osgeo import gdal
gdal.UseExceptions()
# 안전을 위해 GDAL에도 설정 반영
gdal.SetConfigOption('CPL_LOG', os.environ.get('CPL_LOG'))
if os.environ.get('GDAL_DRIVER_PATH'):
    gdal.SetConfigOption('GDAL_DRIVER_PATH', os.environ['GDAL_DRIVER_PATH'])
if os.environ.get('GDAL_DATA'):
    gdal.SetConfigOption('GDAL_DATA', os.environ['GDAL_DATA'])

# 입출력 디렉토리
IN_DIR = r"D:\Landslide\data\LDAPS\NetCDF"
OUT_DIR = r"D:\Landslide\data\LDAPS\ldaps_tif_geoloc"

# 출력 디렉토리 생성
os.makedirs(os.path.join(OUT_DIR, "acc5d"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "peak1h"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)

def validate_precipitation_variable(ds):
    """precipitation 변수 명시적 선택 및 검증"""
    
    # 1) 정확한 변수명으로 찾기
    if 'precipitation' in ds.data_vars:
        var_name = 'precipitation'
        print(f"[INFO] Found explicit 'precipitation' variable")
    else:
        # 2) 강수량 관련 변수 찾기
        precip_candidates = [v for v in ds.data_vars.keys() 
                           if any(p in v.lower() for p in ['precip', 'rain', 'lspr'])]
        
        if precip_candidates:
            var_name = precip_candidates[0]
            print(f"[INFO] Selected precipitation candidate: {var_name}")
        else:
            raise ValueError("No precipitation variable found in dataset")
    
    precip_var = ds[var_name]
    
    # 3) 메타데이터 확인
    units = precip_var.attrs.get('units', 'unknown')
    long_name = precip_var.attrs.get('long_name', 'precipitation')
    
    print(f"[INFO] Variable metadata: units={units}, long_name={long_name}")
    
    # 4) 단위 변환 필요성 확인
    if units == 'kg m-2 s-1':
        print(f"[INFO] Converting units: kg m-2 s-1 → mm/h")
        precip_var = precip_var * 3600.0
        precip_var.attrs['units'] = 'mm/h'
        precip_var.attrs['conversion_note'] = 'Converted from kg m-2 s-1'
    
    return precip_var

def validate_coordinates(ds):
    """2D 커브리니어 좌표 검증"""
    
    if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
        raise ValueError("Required 2D latitude/longitude coordinates not found")
    
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    if lats.ndim != 2 or lons.ndim != 2:
        raise ValueError(f"Expected 2D coordinates, got lat: {lats.ndim}D, lon: {lons.ndim}D")
    
    if lats.shape != lons.shape:
        raise ValueError(f"Coordinate shape mismatch: lat {lats.shape} vs lon {lons.shape}")
    
    print(f"[INFO] Validated curvilinear coordinates: {lats.shape}")
    print(f"[INFO] Coordinate ranges: lat {lats.min():.3f}~{lats.max():.3f}, lon {lons.min():.3f}~{lons.max():.3f}")
    
    return lats, lons

def compute_daily_statistics(precip_data, day_str):
    """일별 통계 계산 with 품질 관리"""
    
    if 'time' not in precip_data.dims:
        print(f"[INFO] Single time step data, using as-is")
        return precip_data, precip_data, {"hours_available": 1, "hours_expected": 1}
    
    # 시간축 품질 분석
    time_size = precip_data.time.size
    valid_data = precip_data.where(np.isfinite(precip_data), 0.0)  # NaN → 0
    
    # 각 시간대별 유효 데이터 픽셀 수
    valid_pixels_per_time = (~np.isnan(precip_data)).sum(dim=['y', 'x'])
    fully_missing_times = (valid_pixels_per_time == 0).sum().item()
    
    quality_info = {
        "hours_expected": 24,
        "hours_available": time_size,
        "hours_fully_missing": fully_missing_times,
        "data_completeness": (time_size - fully_missing_times) / 24.0
    }
    
    # 일별 집계
    sum_day = valid_data.sum(dim="time", skipna=True)
    sum_day.attrs.update({
        'long_name': 'Daily accumulated precipitation',
        'units': 'mm',
        'aggregation_method': 'sum over time dimension',
        'quality': f"{quality_info['data_completeness']:.1%} complete"
    })
    
    peak_day = valid_data.max(dim="time", skipna=True)
    peak_day.attrs.update({
        'long_name': 'Daily peak precipitation rate', 
        'units': 'mm/h',
        'aggregation_method': 'max over time dimension',
        'quality': f"{quality_info['data_completeness']:.1%} complete"
    })
    
    print(f"[INFO] Time quality: {time_size}/24 hours, {quality_info['data_completeness']:.1%} complete")
    
    return sum_day, peak_day, quality_info

def create_geolocation_netcdf(data_array, lats, lons, temp_path, var_name):
    """지오로케이션 워핑용 임시 NetCDF 생성"""
    
    height, width = data_array.shape
    
    # 새로운 Dataset 생성 (geolocation arrays 포함)
    ds_temp = xr.Dataset(
        data_vars={
            var_name: (("y", "x"), data_array.values.astype(np.float32), data_array.attrs),
            "latitude": (("y", "x"), lats.astype(np.float32), 
                        {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": (("y", "x"), lons.astype(np.float32),
                         {"standard_name": "longitude", "units": "degrees_east"})
        },
        coords={
            "y": np.arange(height),
            "x": np.arange(width)
        },
        attrs={
            "Conventions": "CF-1.8",
            "geospatial_info": "2D curvilinear coordinates for geolocation warping"
        }
    )
    
    # coordinates 속성 추가 (GDAL geoloc 힌트)
    ds_temp[var_name].attrs["coordinates"] = "latitude longitude"
    
    # NetCDF 저장
    ds_temp.to_netcdf(temp_path, encoding={
        var_name: {"zlib": True, "complevel": 4, "dtype": "float32"},
        "latitude": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "longitude": {"zlib": True, "complevel": 4, "dtype": "float32"}
    })
    
    ds_temp.close()
    print(f"[INFO] Created geolocation NetCDF: {os.path.basename(temp_path)}")

def gdal_geoloc_warp(temp_nc, var_name, output_tif, target_res=0.015):
    """GDAL 지오로케이션 워핑으로 정확한 GeoTIFF 생성"""
    
    # 디버깅: NetCDF 파일이 존재하는지 확인
    if not os.path.exists(temp_nc):
        raise RuntimeError(f"Temporary NetCDF file does not exist: {temp_nc}")
    
    print(f"[DEBUG] Checking NetCDF file: {temp_nc}")
    
    # 1차: 변수 서브데이터셋을 직접 지정하여 열기
    target_subdataset = f'NETCDF:"{temp_nc}":{var_name}'
    print(f"[DEBUG] Attempting to open: {target_subdataset}")
    try:
        src_ds = gdal.Open(target_subdataset, gdal.GA_ReadOnly)
    except Exception as e:
        print(f"[ERROR] Exception during gdal.Open: {e}")
        src_ds = None
    
    # 실패 시: 디버깅을 위해 컨테이너 열어서 서브데이터셋 목록 출력 후 재시도
    if src_ds is None:
        print(f"[WARN] Direct open failed: {target_subdataset}")
        nc_ds = gdal.Open(temp_nc, gdal.GA_ReadOnly)
        if nc_ds is None:
            last_err = gdal.GetLastErrorMsg()
            raise RuntimeError(f"Cannot open NetCDF file: {temp_nc} | {last_err}")
        subdatasets = nc_ds.GetSubDatasets()
        print(f"[DEBUG] Available subdatasets:")
        for i, (name, desc) in enumerate(subdatasets):
            print(f"  [{i}] {name} -> {desc}")
        nc_ds = None
        # fallback 선택
        for name, desc in subdatasets:
            if var_name in name:
                target_subdataset = name
                print(f"[INFO] Found matching subdataset: {name}")
                break
        else:
            if subdatasets:
                target_subdataset = subdatasets[0][0]
                print(f"[WARNING] Variable '{var_name}' not found, using first subdataset: {target_subdataset}")
            else:
                raise RuntimeError(f"No subdatasets found in {temp_nc}")
        src_ds = gdal.Open(target_subdataset, gdal.GA_ReadOnly)
    
    if src_ds is None:
        last_err = gdal.GetLastErrorMsg()
        raise RuntimeError(f"Failed to open NetCDF subdataset: {target_subdataset} | {last_err}")
    
    print(f"[INFO] Opened source: {src_ds.RasterXSize}x{src_ds.RasterYSize}")
    
    # Warp 옵션 설정
    warp_options = gdal.WarpOptions(
        # PROJ DB 충돌 회피: EPSG 대신 PROJ4 문자열 사용
        dstSRS="+proj=longlat +datum=WGS84 +no_defs",
        geoloc=True,           # 지오로케이션 배열 사용
        multithread=True,      # 멀티스레드
        xRes=target_res,       # 목표 해상도 (도)
        yRes=target_res,
        outputType=gdal.GDT_Float32,
        srcNodata=np.nan,
        dstNodata=np.nan,
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=2", 
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "BIGTIFF=IF_NEEDED"
        ]
    )
    
    # Warp 수행
    print(f"[INFO] Starting GDAL Warp to {os.path.basename(output_tif)}")
    result_ds = gdal.Warp(output_tif, src_ds, options=warp_options)
    
    if result_ds is None:
        last_err = gdal.GetLastErrorMsg()
        raise RuntimeError(f"GDAL Warp failed for {output_tif} | {last_err}")
    
    # 결과 정보
    print(f"[INFO] Output GeoTIFF: {result_ds.RasterXSize}x{result_ds.RasterYSize}")
    
    # 리소스 해제
    src_ds = None
    result_ds = None

def main():
    """메인 처리 함수"""
    print(f"[INFO] Starting LDAPS Geolocation Warping Pipeline")
    print(f"[INFO] Input: {IN_DIR}")
    print(f"[INFO] Output: {OUT_DIR}")
    
    # NetCDF 파일 목록
    files = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".nc")])
    if not files:
        raise SystemExit(f"No NetCDF files found in {IN_DIR}")
    
    print(f"[INFO] Found {len(files)} NetCDF files")
    
    rolling = deque(maxlen=5)  # 5일 롤링 윈도우
    daily_sums = {}
    processing_log = []

    # 각 파일 처리
    for i, filename in enumerate(files):
        filepath = os.path.join(IN_DIR, filename)
        
        try:
            # 날짜 추출
            day_str = filename.replace("LDAPS_Rain_", "").replace(".nc", "")
            print(f"\n[PROCESS] {day_str} ({i+1}/{len(files)})")
            
            # NetCDF 열기
            with xr.open_dataset(filepath) as ds:
                
                print(f"[INFO] Dataset: {list(ds.data_vars.keys())}")
                print(f"[INFO] Conventions: {ds.attrs.get('Conventions', 'None')}")
                
                # 1) precipitation 변수 검증 및 선택
                precip_data = validate_precipitation_variable(ds)
                
                # 2) 커브리니어 좌표 검증
                lats, lons = validate_coordinates(ds)
                
                # 3) 일별 통계 계산
                sum_day, peak_day, quality_info = compute_daily_statistics(precip_data, day_str)
                
                # 로그 기록
                processing_log.append({
                    'date': day_str,
                    'quality': quality_info,
                    'sum_max': float(sum_day.max().values) if not np.isnan(sum_day.max().values) else 0.0,
                    'peak_max': float(peak_day.max().values) if not np.isnan(peak_day.max().values) else 0.0
                })
                
                # 4) 5일 롤링
                daily_sums[day_str] = sum_day
                rolling.append(sum_day)
                
                if len(rolling) > 1:
                    acc5 = xr.concat(list(rolling), dim="rolling_day").sum(dim="rolling_day", skipna=True)
                    acc5.attrs.update({
                        'long_name': '5-day accumulated precipitation',
                        'units': 'mm',
                        'rolling_days': len(rolling)
                    })
                else:
                    acc5 = sum_day
                
                # 5) 임시 NetCDF 생성 및 GDAL 워핑 (Windows 파일 잠금 회피를 위해 mkstemp 사용)
                fd_acc, path_acc = tempfile.mkstemp(suffix='.nc')
                os.close(fd_acc)
                fd_peak, path_peak = tempfile.mkstemp(suffix='.nc')
                os.close(fd_peak)
                try:
                    # 5일 누적용 임시 NetCDF
                    create_geolocation_netcdf(acc5, lats, lons, path_acc, "precipitation_acc5d")
                    
                    # 일피크용 임시 NetCDF
                    create_geolocation_netcdf(peak_day, lats, lons, path_peak, "precipitation_peak1h")
                    
                    # GDAL 워핑으로 최종 GeoTIFF 생성
                    out_acc = os.path.join(OUT_DIR, "acc5d", f"{day_str}_acc5d_mm_geoloc.tif")
                    out_peak = os.path.join(OUT_DIR, "peak1h", f"{day_str}_peak1h_mm_geoloc.tif")
                    
                    gdal_geoloc_warp(path_acc, "precipitation_acc5d", out_acc)
                    gdal_geoloc_warp(path_peak, "precipitation_peak1h", out_peak)
                finally:
                    # 임시 파일 삭제 (존재하면)
                    with contextlib.suppress(Exception):
                        os.unlink(path_acc)
                    with contextlib.suppress(Exception):
                        os.unlink(path_peak)
                
                print(f"[SAVE] {os.path.basename(out_acc)} (max: {processing_log[-1]['sum_max']:.1f} mm)")
                print(f"[SAVE] {os.path.basename(out_peak)} (max: {processing_log[-1]['peak_max']:.1f} mm/h)")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 로그 기록
            processing_log.append({
                'date': day_str,
                'error': str(e),
                'quality': None
            })
            continue
    
    # 처리 로그 저장
    log_df = pd.DataFrame(processing_log)
    log_file = os.path.join(OUT_DIR, "logs", "processing_log.csv")
    log_df.to_csv(log_file, index=False)
    
    print(f"\n[COMPLETE] Processed {len([l for l in processing_log if 'error' not in l])} days successfully!")
    print(f"[OUTPUT] Files saved to: {OUT_DIR}")
    print(f"[LOG] Processing log saved to: {log_file}")

if __name__ == "__main__":
    main()