# ldaps_download_grib_regions.py
# ------------------------------------------------------------
# GRIB 파일 다운로드 후 NetCDF로 변환하여 지역별 강수량 데이터 처리
# Requires: requests, pandas, numpy, xarray, geopandas, shapely, pyproj, rioxarray, rasterio, cfgrib
# Install (conda): conda install -y requests pandas numpy xarray geopandas shapely pyproj rioxarray rasterio cfgrib eccodes
# ------------------------------------------------------------
import os, time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import requests
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.ops import unary_union
from rasterio.features import rasterize
import rasterio
from rasterio.transform import Affine
from pyproj import CRS

# GRIB 파일 다운로드 API 엔드포인트
GRIB_API_BASE = "https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php"

# LCC 좌표계 정의
LCC_CRS = CRS.from_proj4("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +datum=WGS84 +units=m +no_defs")

def download_grib_file(service_key: str, base_time: str, lead_hour: int, variable: str, output_dir: str) -> str:
    """GRIB 파일을 다운로드하고 파일 경로를 반환"""
    # URL 생성 (ef 파라미터는 lead_hour+1)
    url = f'{GRIB_API_BASE}?nwp=l015&sub=unis&vars={variable}&tmfc={base_time}&ef={lead_hour+1}&dataType=GRIB&authKey={service_key}'
    
    # 파일명 생성
    filename = f"l015_{base_time}_{variable}_h{lead_hour:03d}.gb2"
    filepath = os.path.join(output_dir, filename)
    
    # 이미 파일이 존재하면 스킵
    if os.path.exists(filepath):
        print(f"[SKIP] {filename} already exists")
        return filepath
    
    print(f"[DOWNLOAD] {filename}")
    
    try:
        r = requests.get(url, timeout=(10, 180), stream=True)
        
        if r.status_code != 200:
            print(f"[ERROR] HTTP {r.status_code} for {filename}: {r.text[:200]}")
            return None
            
        # 응답 헤더 확인
        content_type = r.headers.get('content-type', '')
        if 'application/octet-stream' not in content_type:
            # 에러 메시지일 수 있음
            error_text = r.text[:200]
            if "file not exist" in error_text or "파라미터 없음" in error_text:
                print(f"[SKIP] {filename}: {error_text.strip()}")
                return None
            
        # GRIB 파일 저장
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        file_size = os.path.getsize(filepath)
        if file_size > 0:
            print(f"[SUCCESS] Downloaded {filename} ({file_size} bytes)")
            return filepath
        else:
            os.remove(filepath)  # 빈 파일 삭제
            return None
        
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return None

def grib_to_xarray(grib_file: str) -> xr.Dataset:
    """GRIB 파일을 xarray Dataset으로 로드"""
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to load GRIB {grib_file}: {e}")
        return None

def select_forecast_times_for_day(day: datetime) -> List[Tuple[str, int]]:
    """하루 24시간을 커버하는 예보시간 조합 생성"""
    # 간단한 전략: 해당 날짜의 00, 06, 12, 18시 초기시간에서 각각 0~5시간 선행시간
    kst = day
    def bt(d, h): return d.strftime("%Y%m%d") + f"{h:02d}00"
    
    calls = []
    for init_hour in [0, 6, 12, 18]:
        for lead in range(0, 6):  # 0~5시간 선행시간
            calls.append((bt(kst, init_hour), lead))
    
    return calls

def union_target_regions(
    admin_shp: str,
    target_names: List[str],
    to_crs: CRS,
    name_col: str = None,
    encoding: str = None
):
    """행정구역 shapefile에서 대상 지역들을 union하여 반환"""
    gdf = gpd.read_file(admin_shp, encoding=encoding) if encoding else gpd.read_file(admin_shp)
    
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    
    # 컬럼명 자동 탐지
    if name_col is None:
        candidates = ["CTP_KOR_NM", "SIDO", "SIG_KOR_NM", "ADM_NM", "adm_nm", "NAME_1", "CTP_ENG_NM"]
        name_col = next((c for c in candidates if c in gdf.columns), None)
        if name_col is None:
            raise ValueError(f"시·도명 컬럼을 찾지 못함. 사용 가능한 컬럼: {list(gdf.columns)}")
    
    # 이름 정규화
    def norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    
    gdf["_name_"] = norm(gdf[name_col])
    target_norm = [n.replace(" ", "") for n in target_names]
    
    # 대상 선택 후 좌표계 변환
    sub = gdf.loc[gdf["_name_"].isin(target_norm)]
    if sub.empty:
        uniq = sorted(gdf["_name_"].unique())[:30]
        raise ValueError(f"선택한 시·도명이 매칭되지 않음: {target_names}\\n예시 값(일부): {uniq}")
    
    sub = sub.to_crs(to_crs)
    return unary_union(sub.geometry)

def make_mask_from_grib(ds: xr.Dataset, geom_union) -> np.ndarray:
    """GRIB 데이터의 좌표에 기반한 마스크 생성"""
    try:
        # 위경도 좌표 가져오기
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            lats = ds.latitude.values
            lons = ds.longitude.values
        else:
            print("[WARNING] No lat/lon coordinates found in GRIB")
            return None
            
        # 좌표를 LCC로 변환하여 마스크 생성
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", LCC_CRS)
        
        # 격자 변환
        x_lcc, y_lcc = transformer.transform(lats.flatten(), lons.flatten())
        x_lcc = x_lcc.reshape(lats.shape)
        y_lcc = y_lcc.reshape(lats.shape)
        
        # 각 격자점이 지역 내부에 있는지 확인
        from shapely.geometry import Point
        mask = np.zeros(lats.shape, dtype=bool)
        
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                point = Point(x_lcc[i, j], y_lcc[i, j])
                mask[i, j] = geom_union.contains(point)
        
        return mask
        
    except Exception as e:
        print(f"[ERROR] Failed to create mask: {e}")
        return None

def main():
    SERVICE_KEY = os.environ.get("KMA_SERVICE_KEY", "5LGEH7S2SVSxhB-0tnlUlA")
    if not SERVICE_KEY:
        raise SystemExit("환경변수 KMA_SERVICE_KEY 가 필요합니다.")
    print(f"[INFO] Using API key: {SERVICE_KEY[:10]}...")
    
    # 원본 기간 유지
    START = "2020-03-11"
    END   = "2020-09-19"
    
    GRIB_DIR = r"D:\Landslide\data\LDAPS\GRIB"  # GRIB 파일 저장
    OUT_DIR = r"D:\Landslide\data\LDAPS\NetCDF"  # NetCDF 파일 저장
    os.makedirs(GRIB_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 행정구역 설정
    ADMIN_SHP = r"D:\Landslide\data\행정구역경계\행정구역_시도.shp"
    TARGET_SIDOS = ["경상남도","대구","경상북도","울산","부산"]
    
    # 지역 경계 union
    geom_union = union_target_regions(
        ADMIN_SHP,
        TARGET_SIDOS,
        LCC_CRS,
        name_col="CTP_ENG_NM",
        encoding=None
    )
    
    # 강수량 변수
    VARIABLES = ["lspr"]  # Large-scale precipitation rate
    
    for day in pd.date_range(START, END, freq="D"):
        print(f"\\n[DAY] Processing {day.date()}")
        
        # 해당 날짜의 예보시간 조합
        forecast_times = select_forecast_times_for_day(day.to_pydatetime())
        
        daily_data = []
        valid_times = []
        
        for base_time, lead_hour in forecast_times:
            for variable in VARIABLES:
                # GRIB 파일 다운로드
                grib_file = download_grib_file(SERVICE_KEY, base_time, lead_hour, variable, GRIB_DIR)
                
                if grib_file is None:
                    continue
                
                try:
                    # GRIB 파일을 xarray로 로드
                    ds = grib_to_xarray(grib_file)
                    if ds is None:
                        continue
                    
                    # 마스크 적용
                    mask = make_mask_from_grib(ds, geom_union)
                    if mask is None:
                        print(f"[WARNING] Could not create mask for {grib_file}")
                        continue
                    
                    # 데이터 변수 식별
                    data_vars = list(ds.data_vars.keys())
                    if not data_vars:
                        print(f"[WARNING] No data variables in {grib_file}")
                        continue
                    
                    data_var = data_vars[0]  # 첫 번째 데이터 변수 사용
                    data_array = ds[data_var].values
                    
                    # 마스크 적용
                    masked_data = np.where(mask, data_array, np.nan)
                    
                    # 유효시간 계산
                    base_dt = datetime.strptime(base_time, "%Y%m%d%H%M")
                    valid_time = base_dt + timedelta(hours=lead_hour)
                    
                    # 해당 날짜에 속하는 시간만 수집
                    if valid_time.date() == day.date():
                        daily_data.append(masked_data)
                        valid_times.append(valid_time)
                    
                    ds.close()  # 메모리 해제
                    
                except Exception as e:
                    print(f"[ERROR] Processing {grib_file}: {e}")
                    continue
                
                # 메모리 절약을 위해 잠시 대기
                time.sleep(0.1)
        
        if not daily_data:
            print(f"[SKIP] No data for {day.date()}")
            continue
        
        # 시간순 정렬
        if len(daily_data) > 1:
            sorted_indices = sorted(range(len(valid_times)), key=lambda i: valid_times[i])
            daily_data = [daily_data[i] for i in sorted_indices]
            valid_times = [valid_times[i] for i in sorted_indices]
        
        # xarray Dataset 생성
        data_stack = np.stack(daily_data, axis=0)
        
        # 시간 차원 생성 (24시간 격자, 누락된 시간은 NaN)
        start_time = pd.Timestamp(day.date())
        full_time_range = pd.date_range(start_time, start_time + pd.Timedelta(hours=23), freq="1H")
        
        # 24시간 배열 초기화
        full_data = np.full((24, data_stack.shape[1], data_stack.shape[2]), np.nan, dtype=np.float32)
        
        # 데이터 매핑
        for i, vt in enumerate(valid_times):
            hour_idx = (vt - start_time).total_seconds() // 3600
            if 0 <= hour_idx < 24:
                full_data[int(hour_idx)] = daily_data[i]
        
        # xarray Dataset 생성
        ds_daily = xr.Dataset(
            {"precipitation": (("time", "y", "x"), full_data)},
            coords={
                "time": full_time_range,
                "y": np.arange(data_stack.shape[1]),
                "x": np.arange(data_stack.shape[2])
            }
        )
        
        # 속성 추가
        ds_daily["precipitation"].attrs.update(
            units="kg m**-2 s**-1",
            long_name="LDAPS precipitation rate (masked for target regions)"
        )
        ds_daily.attrs.update(
            regions=", ".join(TARGET_SIDOS),
            crs=str(LCC_CRS.to_wkt())
        )
        
        # NetCDF 파일로 저장
        output_file = os.path.join(OUT_DIR, f"LDAPS_Rain_{day.strftime('%Y%m%d')}.nc")
        ds_daily.to_netcdf(output_file)
        print(f"[SAVE] {output_file}")
        
        ds_daily.close()

if __name__ == "__main__":
    main()