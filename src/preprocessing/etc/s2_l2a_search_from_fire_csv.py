# s2_l2a_search_from_fire_csv.py
"""
실행 가상환경: qgispy

예시 실행:
python s2_l2a_search_from_fire_csv.py ^
  --input "D:\\Landslide\\data\\산불발생이력\\wildfire_경남_20151001_20231231_extracted.csv" ^
  --output "D:/Landslide/data/산불발생이력/s2_matches.csv" ^
  --buffer-days 15 ^
  --cloud-threshold 40 ^
  --bbox-km 1 ^
  --n05-only

목적:
  산불 진화일(End_DT)로부터 +buffer-days 동안의 Sentinel-2 L2A 장면을
  Copernicus Data Space STAC(v1)에서 검색하고,
  (진화일, 촬영일, 제품명(SAFE), 구름율) 페어로 CSV 저장.

요구 패키지:
  pip install pystac-client shapely pandas python-dateutil

입력 CSV 최소 컬럼:
  - End_DT : 산불 진화일 (예: '2021-05-12' 또는 파싱 가능한 문자열)
  - Lat_DD : 위도 (WGS84, 소수)
  - Lon_DD : 경도 (WGS84, 소수)
  - Damage_BA : 산불 피해면적 (선택적)

옵션:
  --buffer-days       : 진화일 이전/이후 검색 기간(일). 기본 15
  --cloud-threshold   : 최대 구름율(%). 기본 50
  --bbox-km           : 점 주변 검색 반경(km)를 bbox로 변환해 사용. 기본 1km
  --best              : (사용하지 않음 - 자동으로 최적 영상 선택)
  --n05-only          : 파일명(item.id)에 '_N05' 포함되는 제품만 저장(최신 Processing Baseline 고정)
  --encoding          : 입력 CSV 인코딩(기본 'utf-8-sig')

출력 CSV 컬럼:
  fire_id, fire_end_date, Damage_BA, s2_pre_date, s2_post_date, pre_product_id, post_product_id, cloud_cover, days_after_fire, days_before_fire, fire_lon, fire_lat
  - fire_id = 산불 고유 식별자 (원본 CSV 행 번호)
  - fire_end_date = 산불 진화일
  - Damage_BA = 산불 피해면적
  - s2_pre_date, s2_post_date = 산불 이전/이후 영상 촬영일
  - pre_product_id, post_product_id = 산불 이전/이후 Copernicus SAFE 제품명
  - cloud_cover = 구름 피복률
  - days_before_fire, days_after_fire = 진화일과 영상촬영일 간의 차이(일)
  - fire_lon, fire_lat = 산불 발생 위치
"""

import argparse
import math
from datetime import timedelta

import pandas as pd
from dateutil import parser as dparser
from shapely.geometry import Point
from pystac_client import Client


CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
S2_L2A_COLLECTION = "sentinel-2-l2a"


def parse_date(x):
    if pd.isna(x):
        return None
    try:
        return dparser.parse(str(x)).date()
    except Exception:
        return None


def km_to_deg(lon, lat, km):
    """
    주어진 경위도에서 반경 km를 위/경도 degree 단위 half-extent로 근사 변환.
    위도 1도 ≈ 111.32 km, 경도 1도 ≈ 111.32 * cos(lat) km.
    반환: (dx_deg, dy_deg)
    """
    lat_rad = math.radians(float(lat))
    dy = km / 111.32
    # 경도는 위도에 따라 수축
    denom = 111.32 * max(math.cos(lat_rad), 1e-6)
    dx = km / denom
    return dx, dy


def main(args):
    # 입력 CSV 로드
    df = pd.read_csv(args.input, encoding=args.encoding)

    # 컬럼 탐색 (대소문자 변형 허용)
    cols = {c.lower(): c for c in df.columns}
    col_date = cols.get("end_dt")
    col_lat = cols.get("lat_dd")
    col_lon = cols.get("lon_dd")
    col_damage = cols.get("damage_ba")  # 선택적 컬럼
    if not all([col_date, col_lat, col_lon]):
        raise SystemExit("CSV must contain columns: End_DT, Lat_DD, Lon_DD (대소문자 무관).")

    # 파싱/정리
    df["__end_date__"] = df[col_date].apply(parse_date)
    df["__lat__"] = pd.to_numeric(df[col_lat], errors="coerce")
    df["__lon__"] = pd.to_numeric(df[col_lon], errors="coerce")
    if col_damage:
        df["__damage_ba__"] = pd.to_numeric(df[col_damage], errors="coerce")
    else:
        df["__damage_ba__"] = None
    df = df.dropna(subset=["__end_date__", "__lat__", "__lon__"]).reset_index(drop=True)

    # Copernicus Data Space STAC v1 클라이언트
    client = Client.open(CDSE_STAC_URL)

    def search_sentinel2_images(fire_idx, end_date, lat, lon, damage_ba, bbox, is_post_fire=True):
        """산불 이전 또는 이후 영상 검색"""
        buffer_days = int(args.buffer_days)

        if is_post_fire:
            # 산불 이후 검색 (기존 로직)
            t0 = end_date
            t1 = end_date + timedelta(days=buffer_days)
        else:
            # 산불 이전 검색 (새로운 로직)
            t0 = end_date - timedelta(days=buffer_days)
            t1 = end_date

        # STAC 검색
        search = client.search(
            collections=[S2_L2A_COLLECTION],
            bbox=bbox,
            datetime=f"{t0.isoformat()}/{t1.isoformat()}",
            limit=1000,
        )

        best_item = None
        best_cloud = float('inf')

        for item in search.items():
            props = item.properties or {}
            cloud = props.get("eo:cloud_cover", props.get("s2:cloudy_pixel_percentage"))
            try:
                cloud_val = float(cloud) if cloud is not None else None
            except Exception:
                cloud_val = None

            # 클라이언트 측 구름율 필터
            if cloud_val is None or cloud_val > float(args.cloud_threshold):
                continue

            # N05 전용 필터(선택)
            if args.n05_only and "_N05" not in item.id:
                continue

            # 가장 좋은 이미지 선택 (구름율 최소)
            if cloud_val < best_cloud:
                best_cloud = cloud_val
                best_item = item

        if best_item:
            props = best_item.properties or {}
            dt_iso = props.get("datetime") or (best_item.datetime.isoformat() if best_item.datetime else None)
            s2_date = dt_iso[:19] if isinstance(dt_iso, str) else None

            # 날짜 차이 계산
            days_diff = None
            if s2_date:
                try:
                    s2_date_obj = pd.to_datetime(s2_date).date()
                    days_diff = (s2_date_obj - end_date).days
                except Exception:
                    pass

            return {
                "s2_date": s2_date,
                "product_id": best_item.id,
                "cloud_cover": best_cloud,
                "days_diff": days_diff
            }
        else:
            return {
                "s2_date": None,
                "product_id": None,
                "cloud_cover": None,
                "days_diff": None
            }

    rows = []
    for fire_idx, r in df.iterrows():
        end_date = r["__end_date__"]
        lat = float(r["__lat__"])
        lon = float(r["__lon__"])
        damage_ba = r["__damage_ba__"] if "__damage_ba__" in r and not pd.isna(r["__damage_ba__"]) else None

        # 점 주변 bbox 구성 (반경 km → degree half-extent)
        dx, dy = km_to_deg(lon, lat, float(args.bbox_km))
        bbox = (lon - dx, lat - dy, lon + dx, lat + dy)

        # 산불 이전 영상 검색
        pre_result = search_sentinel2_images(fire_idx, end_date, lat, lon, damage_ba, bbox, is_post_fire=False)

        # 산불 이후 영상 검색
        post_result = search_sentinel2_images(fire_idx, end_date, lat, lon, damage_ba, bbox, is_post_fire=True)

        # 결과 행 생성
        rows.append({
            "fire_id": fire_idx,
            "fire_end_date": end_date.isoformat(),
            "Damage_BA": damage_ba,
            "s2_pre_date": pre_result["s2_date"],
            "s2_post_date": post_result["s2_date"],
            "pre_product_id": pre_result["product_id"],
            "post_product_id": post_result["product_id"],
            "cloud_cover": post_result["cloud_cover"] if post_result["cloud_cover"] is not None else pre_result["cloud_cover"],
            "days_after_fire": post_result["days_diff"],
            "days_before_fire": abs(pre_result["days_diff"]) if pre_result["days_diff"] is not None else None,
            "fire_lon": lon,
            "fire_lat": lat,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No matching items found.")
        out.to_csv(args.output, index=False)
        return

    # 정렬: 산불ID 순으로
    out = out.sort_values(by=["fire_id"], ascending=[True])

    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} rows -> {args.output}")

    # 통계 출력
    pre_found = out["pre_product_id"].notna().sum()
    post_found = out["post_product_id"].notna().sum()
    total_fires = len(out)

    print(f"Statistics:")
    print(f"  Total fires: {total_fires}")
    print(f"  Pre-fire images found: {pre_found} ({pre_found/total_fires*100:.1f}%)")
    print(f"  Post-fire images found: {post_found} ({post_found/total_fires*100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to wildfire CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument("--buffer-days", type=int, default=15, help="Days before/after fire end date to search (default: 15)")
    ap.add_argument("--cloud-threshold", type=float, default=50.0, help="Max cloud cover percent (default: 50)")
    ap.add_argument("--bbox-km", type=float, default=1.0, help="Search radius in km around (lon,lat) (default: 1.0)")
    ap.add_argument("--best", action="store_true", help="(Deprecated) Automatically selects best images")
    ap.add_argument("--n05-only", action="store_true", help="Keep only items with '_N05' in SAFE name (Collection-1)")
    ap.add_argument("--encoding", default="utf-8-sig", help="Input CSV encoding (default: utf-8-sig)")
    main(ap.parse_args())
