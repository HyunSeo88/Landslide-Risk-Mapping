# ldaps_load_grib.py
# ------------------------------------------------------------
# GRIB 파일 다운로드 후 NetCDF로 변환하여 전체 LDAPS 강수량 데이터 처리
# --- 핵심 변경점 요약 ---
# 1) '해당 일자'의 첫 성공 GRIB에서 lat/lon을 직접 추출(비결정성 제거)
# 2) NetCDF에 CF-Conventions 스타일의 grid_mapping 변수(crs_wgs84) 추가
# 3) NetCDF 변수/좌표에 압축(zlib), 청크(chunk) 지정
# 4) 타임존 플래그와 day 경계 안전장치(옵션) 추가
# ------------------------------------------------------------
import os, time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple
import requests
import numpy as np
import pandas as pd
import xarray as xr
import warnings

# cfgrib timedelta 경고 억제
warnings.filterwarnings('ignore', category=FutureWarning, module='cfgrib')

# GRIB 파일 다운로드 API 엔드포인트
GRIB_API_BASE = "https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php"

# 타임존 설정
KST = timezone(timedelta(hours=9))  # 필요 시 활성화
USE_KST = True                      # False면 naive 유지

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

def grib_to_xarray(grib_file: str):
    """GRIB 파일을 xarray Dataset으로 로드"""
    try:
        ds = xr.open_dataset(grib_file, engine='cfgrib')
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to load GRIB {grib_file}: {e}")
        return None

def pick_reference_latlon_from_ds(ds):
    """당일 처리 중 '처음 성공적으로 로드된 GRIB'의 위경도 2D를 바로 사용"""
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        assert lat.ndim == 2 and lon.ndim == 2, "lat/lon must be 2-D"
        return lat, lon
    # cfgrib이 다른 이름을 쓰는 경우 대비
    for cand_lat, cand_lon in [('lat', 'lon'), ('Latitude', 'Longitude')]:
        if cand_lat in ds and cand_lon in ds:
            return ds[cand_lat].values, ds[cand_lon].values
    raise RuntimeError("No 2-D lat/lon found in GRIB dataset")

def build_daily_netcdf(day: pd.Timestamp, stack, hours, lat2d, lon2d, out_path: str):
    """개선된 NetCDF 생성 함수 - CF 규약 준수, 압축/청킹 적용"""
    # 시간 좌표(24h)
    start = pd.Timestamp(day.date())
    if USE_KST:
        start = start.tz_localize(KST)
        # hours가 이미 timezone 정보를 가지고 있는지 확인
        processed_hours = []
        for h in hours:
            if hasattr(h, 'tzinfo') and h.tzinfo is not None:
                # 이미 timezone이 있으면 KST로 변환
                processed_hours.append(h.astimezone(KST) if h.tzinfo != KST else h)
            else:
                # timezone이 없으면 KST로 localize
                processed_hours.append(pd.Timestamp(h).tz_localize(KST))
        hours = processed_hours
    full_time = pd.date_range(start, start + pd.Timedelta(hours=23), freq="1h", tz=start.tz)

    ny, nx = stack.shape[1], stack.shape[2]
    data_full = np.full((24, ny, nx), np.nan, np.float32)
    
    # 수집된 시간들을 정렬해 매핑
    order = np.argsort(hours)
    for i in order:
        t = hours[i]
        idx = int((t - start).total_seconds() // 3600)
        if 0 <= idx < 24:
            data_full[idx] = stack[i]

    # NetCDF 저장을 위해 시간을 UTC로 변환 (timezone 정보 제거)
    if full_time.tz is not None:
        full_time_utc = full_time.tz_convert('UTC').tz_localize(None)
        time_attrs = {
            "standard_name": "time",
            "long_name": "time", 
            "timezone": "UTC (converted from KST)",
            "original_timezone": "KST"
        }
    else:
        full_time_utc = full_time
        time_attrs = {"standard_name": "time", "long_name": "time"}

    # 좌표/변수 구성
    ds = xr.Dataset(
        data_vars={
            "precipitation": (("time", "y", "x"), data_full,
                              {"units": "kg m-2 s-1", "long_name": "LDAPS precipitation rate",
                               "grid_mapping": "crs_wgs84"})
        },
        coords={
            "time": (("time",), full_time_utc, time_attrs),
            "y": np.arange(ny),
            "x": np.arange(nx),
            # CF 권고: lat/lon을 변수로도 보존(좌표로만 두어도 무방하지만 가독성↑)
            "latitude": (("y", "x"), lat2d, {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": (("y", "x"), lon2d, {"standard_name": "longitude", "units": "degrees_east"}),
        },
        attrs={
            "source": "KMA LDAPS (via cfgrib)",
            "Conventions": "CF-1.8",
            "title": f"LDAPS precipitation for {day.strftime('%Y-%m-%d')} (full grid)",
        }
    )

    # grid_mapping 변수: 위경도 좌표계를 명시 (후속 파이프라인이 인지 가능)
    ds["crs_wgs84"] = xr.DataArray(0, attrs={
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563
    })

    # 압축/청크 옵션
    comp = dict(zlib=True, complevel=4)
    encoding = {
        "precipitation": dict(chunksizes=(1, min(256, ny), min(256, nx)), **comp, dtype="float32"),
        "latitude":      dict(chunksizes=(min(256, ny), min(256, nx)), **comp, dtype="float32"),
        "longitude":     dict(chunksizes=(min(256, ny), min(256, nx)), **comp, dtype="float32"),
    }

    ds.to_netcdf(out_path, encoding=encoding)
    print(f"[SAVE] {out_path}")
    ds.close()

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



def main():
    SERVICE_KEY = os.environ.get("KMA_SERVICE_KEY", "5LGEH7S2SVSxhB-0tnlUlA")
    if not SERVICE_KEY:
        raise SystemExit("환경변수 KMA_SERVICE_KEY 가 필요합니다.")
    print(f"[INFO] Using API key: {SERVICE_KEY[:10]}...")
    
    # 처리할 날짜 범위
    START = "2020-03-11"
    END   = "2020-09-19"
    
    GRIB_DIR = r"D:\Landslide\data\LDAPS\GRIB"  # GRIB 파일 저장
    OUT_DIR = r"D:\Landslide\data\LDAPS\NetCDF"  # NetCDF 파일 저장
    os.makedirs(GRIB_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 강수량 변수
    VARIABLES = ["lspr"]  # Large-scale precipitation rate
    
    for day in pd.date_range(START, END, freq="D"):
        print(f"\\n[DAY] Processing {day.date()}")
        
        # 해당 날짜의 예보시간 조합
        forecast_times = select_forecast_times_for_day(day.to_pydatetime())
        
        daily_data = []
        valid_times = []
        reference_lat2d = None
        reference_lon2d = None
        
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
                    
                    # 첫 번째 성공 GRIB에서 lat/lon 추출 (비결정성 제거)
                    if reference_lat2d is None:
                        try:
                            reference_lat2d, reference_lon2d = pick_reference_latlon_from_ds(ds)
                            print(f"[INFO] Reference coordinates extracted from {grib_file}")
                            print(f"[INFO] Grid shape: {reference_lat2d.shape}")
                            print(f"[INFO] Lat: {reference_lat2d.min():.3f}~{reference_lat2d.max():.3f}")
                            print(f"[INFO] Lon: {reference_lon2d.min():.3f}~{reference_lon2d.max():.3f}")
                        except Exception as e:
                            print(f"[WARNING] Could not extract reference coordinates: {e}")
                    
                    # 데이터 변수 식별
                    data_vars = list(ds.data_vars.keys())
                    if not data_vars:
                        print(f"[WARNING] No data variables in {grib_file}")
                        ds.close()
                        continue
                    
                    data_var = data_vars[0]  # 첫 번째 데이터 변수 사용
                    data_array = ds[data_var].values
                    
                    # 유효시간 계산
                    base_dt = datetime.strptime(base_time, "%Y%m%d%H%M")
                    if USE_KST:
                        base_dt = base_dt.replace(tzinfo=KST)
                    valid_time = base_dt + timedelta(hours=lead_hour)
                    
                    # 해당 날짜에 속하는 시간만 수집
                    if valid_time.date() == day.date():
                        daily_data.append(data_array)
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
            
        if reference_lat2d is None:
            print(f"[ERROR] No reference coordinates found for {day.date()}")
            continue
        
        # 데이터 스택 생성
        data_stack = np.stack(daily_data, axis=0)
        print(f"[INFO] Collected {len(daily_data)} time steps for {day.date()}")
        
        # NetCDF 생성 및 저장
        output_file = os.path.join(OUT_DIR, f"LDAPS_Rain_{day.strftime('%Y%m%d')}.nc")
        build_daily_netcdf(day, data_stack, valid_times, reference_lat2d, reference_lon2d, output_file)

if __name__ == "__main__":
    main()