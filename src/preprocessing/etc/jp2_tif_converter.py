#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel-2 SAFE to TIFF Converter
Copernicus Sentinel-2 L1C/L2A SAFE 파일을 다중 밴드 TIFF로 변환하는 도구
SCL(Scene Classification Layer) 밴드를 포함한 고급 처리 기능 제공

📋 기본 사용법:
    # 명령행 사용
    python jp2_tif_converter.py "S2A_파일.zip" basic         # RGB+NIR
    python jp2_tif_converter.py "S2A_파일.zip" vegetation    # 식생 분석용
    python jp2_tif_converter.py "S2A_파일.zip" scl           # RGB+NIR+SCL (L2A 전용)
    python jp2_tif_converter.py "S2A_파일.zip" classification # 전체 밴드+SCL
    python jp2_tif_converter.py "S2A_파일.zip" hand B02,B03,B04,SCL  # 커스텀

    # 일괄 처리
    python jp2_tif_converter.py "디렉터리" batch scl

🔧 파이썬 API 사용:
    from jp2_tif_converter import *

    # 기본 변환
    convert_rgb_nir("파일.zip")                    # RGB+NIR
    convert_vegetation_bands("파일.zip")           # 식생 분석용

    # SCL 포함 (L2A 전용)
    convert_with_scl("파일.zip")                   # RGB+NIR+SCL
    convert_classification_bands("파일.zip")       # 전체 밴드+SCL
    convert_only_scl("파일.zip")                   # SCL만 추출

    # 고급 설정
    convert_sentinel2_to_tiff("파일.zip",
                             bands=["B02","B03","B04","SCL"],
                             target_resolution=10)

🎯 주요 응용 분야:
    - 토지피복 분류: classification 모드 (전체 밴드+SCL)
    - 식생 분석: vegetation 모드
    - 품질 평가: scl_only 모드 (구름/그림자 마스킹)
    - RGB 시각화: basic 모드

📚 SCL 값별 의미:
    0=분류없음, 1=포화/결함, 2=짙은그림자, 3=구름그림자
    4=식생, 5=비식생, 6=물, 7=구름(저신뢰)
    8=구름(중신뢰), 9=구름(고신뢰), 10=얇은권운, 11=눈/얼음

필요 패키지:
    pip install rasterio numpy
"""

import os
import glob
import zipfile
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import re

def convert_sentinel2_to_tiff(safe_path, output_path=None, bands=None, target_resolution=10):
    """
    Sentinel-2 SAFE 파일을 다중 밴드 TIFF로 변환하는 함수
    
    Parameters:
    safe_path (str): SAFE 파일 또는 압축파일 경로
    output_path (str): 출력 TIFF 파일 경로 (기본값: 입력파일명_converted.tif)
    bands (list): 변환할 밴드 목록 (기본값: ['B02', 'B03', 'B04', 'B08'])
    target_resolution (int): 목표 해상도 (10m 또는 20m)
    
    Returns:
    str: 생성된 TIFF 파일 경로
    """
    
    # 기본 밴드 설정
    if bands is None:
        bands = ['B02', 'B03', 'B04', 'B08']  # RGB + NIR
    
    print(f"=== Sentinel-2 SAFE to TIFF 변환 시작 ===")
    print(f"입력 경로: {safe_path}")
    print(f"변환할 밴드: {bands}")
    print(f"목표 해상도: {target_resolution}m")
    
    # 1. ZIP 파일인지 확인하고 압축 해제
    if safe_path.endswith('.zip'):
        print(f"\n압축 파일 해제 중: {safe_path}")
        extract_dir = os.path.dirname(safe_path)
        with zipfile.ZipFile(safe_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 압축 해제된 SAFE 폴더 찾기
        zip_basename = os.path.basename(safe_path).replace('.zip', '')

        # 파일명에 이미 .SAFE가 포함되어 있는지 확인
        if zip_basename.endswith('.SAFE'):
            safe_name = zip_basename  # 이미 .SAFE가 있으면 그대로 사용
        else:
            safe_name = zip_basename + '.SAFE'  # 없으면 추가

        safe_folder = os.path.join(extract_dir, safe_name)
        print(f"압축 해제 완료: {safe_folder}")

        # 만약 해당 폴더가 없다면 실제 압축 해제된 폴더를 찾아보기
        if not os.path.exists(safe_folder):
            print(f"예상 경로에 폴더가 없습니다. 압축 해제된 폴더를 검색 중...")
            extracted_items = os.listdir(extract_dir)
            safe_folders = [item for item in extracted_items if item.endswith('.SAFE') and os.path.isdir(os.path.join(extract_dir, item))]

            if safe_folders:
                safe_folder = os.path.join(extract_dir, safe_folders[0])
                print(f"발견된 SAFE 폴더: {safe_folder}")
            else:
                print(f"압축 해제된 내용: {extracted_items}")
                raise FileNotFoundError(f"SAFE 폴더를 찾을 수 없습니다. 압축 해제된 내용을 확인해주세요.")
    else:
        safe_folder = safe_path
    
    if not os.path.exists(safe_folder):
        raise FileNotFoundError(f"SAFE 폴더를 찾을 수 없습니다: {safe_folder}")
    
    print(f"SAFE 폴더 경로: {safe_folder}")
    
    # 2. GRANULE 폴더에서 IMG_DATA 경로 찾기
    granule_pattern = os.path.join(safe_folder, "GRANULE", "*")
    granule_dirs = glob.glob(granule_pattern)
    
    if not granule_dirs:
        raise FileNotFoundError(f"GRANULE 폴더를 찾을 수 없습니다: {granule_pattern}")
    
    granule_path = granule_dirs[0]
    img_data_path = os.path.join(granule_path, "IMG_DATA")
    
    print(f"GRANULE 경로: {granule_path}")
    print(f"IMG_DATA 경로: {img_data_path}")
    
    # 3. 해상도별 폴더 구조 확인 (L2A의 경우)
    resolution_folders = {
        10: "R10m",
        20: "R20m",
        60: "R60m"
    }
    
    target_folder = os.path.join(img_data_path, resolution_folders[target_resolution])
    if not os.path.exists(target_folder):
        # L1C 제품인 경우 해상도 폴더가 없을 수 있음
        target_folder = img_data_path
        print(f"해상도별 폴더가 없습니다. 기본 IMG_DATA 폴더 사용: {target_folder}")
    else:
        print(f"해상도별 이미지 데이터 경로: {target_folder}")
    
    # 4. 밴드 파일 경로 수집
    band_files = []
    band_names = []

    # 밴드별 기본 해상도 정의
    band_resolutions = {
        'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10, 'B05': 20, 'B06': 20,
        'B07': 20, 'B08': 10, 'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20, 'B12': 20,
        'SCL': 20  # Scene Classification Layer (L2A 제품에만 포함)
    }

    for band in bands:
        # 밴드의 원본 해상도 확인
        native_resolution = band_resolutions.get(band, target_resolution)

        # 여러 해상도 폴더에서 검색
        search_folders = []

        # 1. 타겟 해상도 폴더
        if os.path.exists(target_folder):
            search_folders.append(target_folder)

        # 2. 밴드의 원본 해상도 폴더
        if native_resolution != target_resolution:
            native_folder = os.path.join(img_data_path, f"R{native_resolution}m")
            if os.path.exists(native_folder):
                search_folders.append(native_folder)

        # 3. 모든 해상도 폴더 (백업)
        for res in [10, 20, 60]:
            res_folder = os.path.join(img_data_path, f"R{res}m")
            if os.path.exists(res_folder) and res_folder not in search_folders:
                search_folders.append(res_folder)

        # 4. 기본 IMG_DATA 폴더 (L1C용)
        if img_data_path not in search_folders:
            search_folders.append(img_data_path)

        files = []
        found_folder = None

        for folder in search_folders:
            # 여러 패턴 시도
            patterns = [
                os.path.join(folder, f"*_{band}_{native_resolution}m.jp2"),  # 원본 해상도
                os.path.join(folder, f"*_{band}_{target_resolution}m.jp2"),  # 타겟 해상도
                os.path.join(folder, f"*_{band}_10m.jp2"),  # 10m
                os.path.join(folder, f"*_{band}_20m.jp2"),  # 20m
                os.path.join(folder, f"*_{band}_60m.jp2"),  # 60m
                os.path.join(folder, f"*_{band}.jp2")  # 해상도 표시 없음
            ]

            # SCL 밴드의 경우 특별한 패턴도 추가 검색
            if band == 'SCL':
                scl_patterns = [
                    os.path.join(folder, f"*_SCL_{native_resolution}m.jp2"),
                    os.path.join(folder, f"*_SCL_{target_resolution}m.jp2"),
                    os.path.join(folder, f"*_SCL_20m.jp2"),  # SCL은 주로 20m
                    os.path.join(folder, f"*_SCL.jp2"),
                    os.path.join(folder, f"*SCL*.jp2")  # 더 광범위한 검색
                ]
                patterns = scl_patterns + patterns

            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    found_folder = folder
                    break

            if files:
                break

        if files:
            band_files.append(files[0])
            band_names.append(band)
            folder_name = os.path.basename(found_folder) if found_folder != img_data_path else "IMG_DATA"
            print(f"✓ 밴드 {band} 파일 발견: {os.path.basename(files[0])} (폴더: {folder_name})")
        else:
            print(f"✗ 경고: 밴드 {band} 파일을 찾을 수 없습니다.")
            print(f"  검색 폴더: {[os.path.basename(f) for f in search_folders]}")
    
    if not band_files:
        raise ValueError("변환할 밴드 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
    
    # 5. 출력 파일명 설정
    if output_path is None:
        safe_name = os.path.basename(safe_folder).replace('.SAFE', '')
        parent_dir = os.path.dirname(safe_folder)
        output_path = os.path.join(parent_dir, f"{safe_name}_multiband_{target_resolution}m.tif")
    
    print(f"\n출력 파일: {output_path}")
    
    # 6. 첫 번째 밴드에서 메타데이터 가져오기
    with rasterio.open(band_files[0]) as src:
        meta = src.meta.copy()
        reference_transform = src.transform
        reference_crs = src.crs
        reference_width = src.width
        reference_height = src.height
        
        meta.update(
            count=len(band_files),
            dtype='uint16',
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512
        )
    
    print(f"참조 해상도: {reference_width} x {reference_height}")
    print(f"좌표계: {reference_crs}")
    
    # 7. 다중 밴드 TIFF 파일 생성
    print(f"\n다중 밴드 TIFF 파일 생성 중...")
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, (band_file, band_name) in enumerate(zip(band_files, band_names), 1):
            print(f"  처리 중: 밴드 {i}/{len(band_files)} - {band_name}")

            with rasterio.open(band_file) as src:
                # 밴드 데이터 읽기
                data = src.read(1)

                # 해상도가 다른 경우 리샘플링
                if (src.width != reference_width or
                    src.height != reference_height or
                    src.transform != reference_transform):

                    print(f"    리샘플링 수행: {src.width}x{src.height} -> {reference_width}x{reference_height}")

                    # 리샘플링을 위한 빈 배열 생성
                    data_resampled = np.empty((reference_height, reference_width), dtype=data.dtype)

                    reproject(
                        data,
                        data_resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=reference_transform,
                        dst_crs=reference_crs,
                        resampling=Resampling.bilinear
                    )
                    dst.write(data_resampled, i)
                else:
                    dst.write(data, i)

                # 밴드 이름 설정 (여러 방법으로 보존)
                dst.set_band_description(i, band_name)

        # 추가적인 메타데이터로 밴드 이름 정보 저장
        band_names_str = ','.join(band_names)
        dst.update_tags(BAND_NAMES=band_names_str)

        # 각 밴드별로도 태그 추가
        for i, band_name in enumerate(band_names, 1):
            dst.update_tags(i, BAND_NAME=band_name,
                           BAND_ID=band_name,
                           DESCRIPTION=f"Sentinel-2 {band_name} band")
    
    print(f"\n=== 변환 완료! ===")
    print(f"출력 파일: {output_path}")
    print(f"포함된 밴드: {', '.join(band_names)}")
    print(f"파일 크기: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return output_path

# 간편 사용 함수들
def convert_rgb_nir(safe_path, output_path=None):
    """RGB + NIR 밴드만 변환 (가장 기본적인 조합)"""
    return convert_sentinel2_to_tiff(safe_path, output_path, ['B02', 'B03', 'B04', 'B08'], 10)

def convert_all_10m_bands(safe_path, output_path=None):
    """모든 10m 해상도 밴드 변환"""
    return convert_sentinel2_to_tiff(safe_path, output_path, ['B02', 'B03', 'B04', 'B08'], 10)

def convert_vegetation_bands(safe_path, output_path=None):
    """식생 분석용 밴드들 변환 (10m로 리샘플링)"""
    vegetation_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    return convert_sentinel2_to_tiff(safe_path, output_path, vegetation_bands, 10)

def convert_water_analysis_bands(safe_path, output_path=None):
    """수질 분석용 밴드들 변환"""
    water_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B08']
    return convert_sentinel2_to_tiff(safe_path, output_path, water_bands, 10)

def convert_with_scl(safe_path, output_path=None, include_all_bands=False):
    """SCL 밴드를 포함한 변환 (L2A 제품 전용)"""
    if include_all_bands:
        # 모든 스펙트럼 밴드 + SCL
        scl_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'SCL']
    else:
        # 기본 RGB+NIR + SCL
        scl_bands = ['B02', 'B03', 'B04', 'B08', 'SCL']
    return convert_sentinel2_to_tiff(safe_path, output_path, scl_bands, 10)

def convert_classification_bands(safe_path, output_path=None):
    """분류 작업에 유용한 밴드들 + SCL"""
    classification_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'SCL']
    return convert_sentinel2_to_tiff(safe_path, output_path, classification_bands, 10)

def convert_only_scl(safe_path, output_path=None):
    """SCL 밴드만 변환 (품질 평가용)"""
    return convert_sentinel2_to_tiff(safe_path, output_path, ['SCL'], 20)

def check_band_names(tiff_path):
    """
    변환된 TIFF 파일의 밴드 이름 정보를 확인하는 함수

    Parameters:
    tiff_path (str): 확인할 TIFF 파일 경로

    Returns:
    dict: 밴드 이름 정보
    """
    print(f"=== 밴드 이름 정보 확인: {os.path.basename(tiff_path)} ===")

    try:
        with rasterio.open(tiff_path) as src:
            band_info = {
                'count': src.count,
                'descriptions': [],
                'tags': src.tags(),
                'band_tags': {}
            }

            # 각 밴드의 description 확인
            for i in range(1, src.count + 1):
                description = src.descriptions[i-1]
                band_tags = src.tags(i)

                band_info['descriptions'].append(description)
                band_info['band_tags'][i] = band_tags

                print(f"밴드 {i}:")
                print(f"  - Description: {description}")
                if band_tags:
                    for key, value in band_tags.items():
                        print(f"  - {key}: {value}")
                print()

            # 전체 파일 태그 확인
            if band_info['tags']:
                print("파일 전체 태그:")
                for key, value in band_info['tags'].items():
                    print(f"  - {key}: {value}")

            return band_info

    except Exception as e:
        print(f"❌ 오류: {e}")
        return None

def get_available_bands(safe_path):
    """SAFE 파일에서 사용 가능한 밴드 목록을 반환"""
    
    # ZIP 파일인 경우 임시 해제
    temp_extracted = False
    if safe_path.endswith('.zip'):
        extract_dir = os.path.dirname(safe_path)
        with zipfile.ZipFile(safe_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        safe_name = os.path.basename(safe_path).replace('.zip', '.SAFE')
        safe_folder = os.path.join(extract_dir, safe_name)
        temp_extracted = True
    else:
        safe_folder = safe_path
    
    try:
        # GRANULE 폴더 찾기
        granule_dirs = glob.glob(os.path.join(safe_folder, "GRANULE", "*"))
        if not granule_dirs:
            return []
        
        img_data_path = os.path.join(granule_dirs[0], "IMG_DATA")
        
        # JP2 파일들 찾기
        jp2_files = glob.glob(os.path.join(img_data_path, "**", "*.jp2"), recursive=True)
        
        # 밴드명 추출
        bands = set()
        for file in jp2_files:
            filename = os.path.basename(file)
            # B02, B03, SCL 등의 패턴 찾기
            parts = filename.split('_')
            for part in parts:
                if part.startswith('B') and (part[1:].isdigit() or part in ['B8A']):
                    bands.add(part)
                elif part == 'SCL':  # Scene Classification Layer
                    bands.add(part)
        
        return sorted(list(bands))
    
    finally:
        # 임시로 해제한 경우 정리 (실제로는 사용자가 수동으로 정리해야 할 수 있음)
        pass

def find_sentinel2_files(directory):
    """
    디렉터리에서 Sentinel-2 파일들을 재귀적으로 찾기

    Parameters:
    directory (str): 검색할 디렉터리 경로

    Returns:
    list: 발견된 Sentinel-2 파일 경로 목록
    """
    sentinel2_files = []

    # Sentinel-2 파일 패턴 (zip 파일과 SAFE 폴더)
    patterns = [
        "**/S2*_MSIL*A_*.zip",        # Sentinel-2 zip 파일
        "**/S2*_MSIL*A_*.SAFE",       # Sentinel-2 SAFE 폴더
    ]

    print(f"=== Sentinel-2 파일 검색 중: {directory} ===")

    for pattern in patterns:
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path, recursive=True)
        sentinel2_files.extend(files)

    # 중복 제거 및 정렬
    sentinel2_files = sorted(list(set(sentinel2_files)))

    print(f"발견된 파일 수: {len(sentinel2_files)}개")
    for i, file in enumerate(sentinel2_files, 1):
        rel_path = os.path.relpath(file, directory)
        print(f"  {i:2d}. {rel_path}")

    return sentinel2_files

def batch_convert_sentinel2(directory, mode="vegetation", custom_bands=None, skip_existing=True, target_resolution=10):
    """
    디렉터리 내의 모든 Sentinel-2 파일을 일괄 변환

    Parameters:
    directory (str): 검색할 디렉터리 경로
    mode (str): 변환 모드 ("basic", "vegetation", "hand")
    custom_bands (list): hand 모드에서 사용할 밴드 목록
    skip_existing (bool): 이미 변환된 파일 건너뛰기
    target_resolution (int): 목표 해상도 (10, 20, 60m)

    Returns:
    dict: 변환 결과 통계
    """

    # Sentinel-2 파일 찾기
    sentinel2_files = find_sentinel2_files(directory)

    if not sentinel2_files:
        print("❌ Sentinel-2 파일을 찾을 수 없습니다.")
        return {"success": 0, "skipped": 0, "failed": 0, "total": 0}

    print(f"\n=== 일괄 변환 시작 (모드: {mode}) ===")

    results = {"success": 0, "skipped": 0, "failed": 0, "total": len(sentinel2_files)}
    failed_files = []

    for i, file_path in enumerate(sentinel2_files, 1):
        print(f"\n[{i}/{len(sentinel2_files)}] 처리 중: {os.path.basename(file_path)}")

        try:
            # 출력 파일명 생성
            if file_path.endswith('.zip'):
                base_name = os.path.basename(file_path).replace('.zip', '')
            else:
                base_name = os.path.basename(file_path).replace('.SAFE', '')

            # 이미 .SAFE가 포함된 경우 제거
            if base_name.endswith('.SAFE'):
                base_name = base_name[:-5]

            output_dir = os.path.dirname(file_path)
            output_file = os.path.join(output_dir, f"{base_name}_multiband_{target_resolution}m.tif")

            # 이미 존재하는 파일 건너뛰기
            if skip_existing and os.path.exists(output_file):
                print(f"  ⏭️  이미 존재함, 건너뛰기: {os.path.basename(output_file)}")
                results["skipped"] += 1
                continue

            # 모드에 따른 변환
            if mode == "basic":
                result = convert_rgb_nir(file_path, output_file)
            elif mode == "vegetation":
                result = convert_vegetation_bands(file_path, output_file)
            elif mode == "scl":
                result = convert_with_scl(file_path, output_file, include_all_bands=False)
            elif mode == "classification":
                result = convert_classification_bands(file_path, output_file)
            elif mode == "scl_only":
                result = convert_only_scl(file_path, output_file)
            elif mode == "hand":
                if not custom_bands:
                    print(f"  ❌ hand 모드에서는 밴드를 지정해야 합니다.")
                    results["failed"] += 1
                    failed_files.append(file_path)
                    continue
                result = convert_sentinel2_to_tiff(file_path, output_file, bands=custom_bands, target_resolution=target_resolution)

            print(f"  ✅ 변환 완료: {os.path.basename(result)}")
            results["success"] += 1

        except Exception as e:
            print(f"  ❌ 변환 실패: {str(e)}")
            results["failed"] += 1
            failed_files.append(file_path)

    # 결과 요약
    print(f"\n=== 일괄 변환 완료 ===")
    print(f"총 파일 수: {results['total']}")
    print(f"변환 성공: {results['success']}")
    print(f"건너뛴 파일: {results['skipped']}")
    print(f"변환 실패: {results['failed']}")

    if failed_files:
        print(f"\n실패한 파일 목록:")
        for file in failed_files:
            print(f"  - {os.path.relpath(file, directory)}")

    return results

def show_usage():
    """사용 방법 출력"""
    print("=== Sentinel-2 SAFE to TIFF Converter ===")
    print()
    print("💡 기본 사용법:")
    print("=================")
    print("단일 파일 변환:")
    print('  python jp2_tif_converter.py "S2A_파일.zip" basic         # RGB+NIR (가장 기본)')
    print('  python jp2_tif_converter.py "S2A_파일.zip" vegetation    # 식생 분석용')
    print('  python jp2_tif_converter.py "S2A_파일.zip" scl           # RGB+NIR+SCL (L2A 전용)')
    print('  python jp2_tif_converter.py "S2A_파일.zip" classification # 전체 밴드+SCL')
    print('  python jp2_tif_converter.py "S2A_파일.zip" scl_only      # SCL만 (품질 확인)')
    print('  python jp2_tif_converter.py "S2A_파일.zip" hand B02,B03,B04,SCL # 커스텀')
    print()
    print("디렉터리 일괄 변환:")
    print('  python jp2_tif_converter.py "디렉터리경로" batch basic')
    print('  python jp2_tif_converter.py "디렉터리경로" batch vegetation')
    print('  python jp2_tif_converter.py "디렉터리경로" batch scl')
    print('  python jp2_tif_converter.py "디렉터리경로" batch classification')
    print('  python jp2_tif_converter.py "디렉터리경로" batch hand B02,B03,B04,SCL')
    print()
    print("🔧 파이썬 함수 사용법:")
    print("======================")
    print("# 기본 변환")
    print('convert_rgb_nir("파일.zip")                    # RGB + NIR')
    print('convert_vegetation_bands("파일.zip")           # 식생 분석용 밴드')
    print('convert_water_analysis_bands("파일.zip")       # 수질 분석용 밴드')
    print()
    print("# SCL 포함 변환 (L2A 제품 전용)")
    print('convert_with_scl("파일.zip")                   # RGB+NIR+SCL')
    print('convert_with_scl("파일.zip", include_all_bands=True)  # 전체 밴드+SCL')
    print('convert_classification_bands("파일.zip")       # 분류용 (전체 밴드+SCL)')
    print('convert_only_scl("파일.zip")                   # SCL 밴드만')
    print()
    print("# 고급 설정")
    print('convert_sentinel2_to_tiff("파일.zip", bands=["B02", "B03", "B04", "SCL"])')
    print('convert_sentinel2_to_tiff("파일.zip", target_resolution=20)  # 20m 해상도')
    print('convert_sentinel2_to_tiff("파일.zip", "출력파일.tif")         # 출력 경로 지정')
    print()
    print("🎯 목적별 추천 조합:")
    print("====================")
    print("📊 토지 분류 작업:")
    print('  python jp2_tif_converter.py "파일.zip" classification')
    print('  # → B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL (10m)')
    print()
    print("🌿 식생 분석:")
    print('  python jp2_tif_converter.py "파일.zip" vegetation')
    print('  # → B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 (10m)')
    print()
    print("🌊 수질 분석:")
    print('  convert_water_analysis_bands("파일.zip")')
    print('  # → B02,B03,B04,B05,B06,B08 (10m)')
    print()
    print("🔍 품질 평가:")
    print('  python jp2_tif_converter.py "파일.zip" scl_only')
    print('  # → SCL만 (20m, 구름/그림자 마스킹용)')
    print()
    print("🎨 RGB 시각화:")
    print('  convert_rgb_nir("파일.zip")')
    print('  # → B02,B03,B04,B08 (10m)')
    print()
    print("🛡️ 고품질 분석 (구름 제거):")
    print('  python jp2_tif_converter.py "파일.zip" scl')
    print('  # → B02,B03,B04,B08,SCL (10m, SCL로 품질 필터링 가능)')
    print()
    print("📋 유틸리티 함수:")
    print("=================")
    print('get_available_bands("파일.zip")               # 사용 가능한 밴드 확인')
    print('check_band_names("변환된파일.tif")            # 밴드 정보 확인')
    print('find_sentinel2_files("디렉터리")              # S2 파일 검색')
    print()
    print("📚 Sentinel-2 밴드 정보:")
    print("=========================")
    print("스펙트럼 밴드:")
    print("  B01: Coastal aerosol (443nm, 60m)   - 대기 보정")
    print("  B02: Blue (490nm, 10m)              - 수심, 구름 감지")
    print("  B03: Green (560nm, 10m)             - 식생 건강도")
    print("  B04: Red (665nm, 10m)               - 클로로필 흡수")
    print("  B05: Red Edge (705nm, 20m)          - 식생 스트레스")
    print("  B06: Red Edge (740nm, 20m)          - 식생 분류")
    print("  B07: Red Edge (783nm, 20m)          - 식생 분류")
    print("  B08: NIR (842nm, 10m)               - 식생량, 수분")
    print("  B8A: Narrow NIR (865nm, 20m)       - 정밀 식생 분석")
    print("  B09: Water vapour (945nm, 60m)     - 대기 수증기")
    print("  B10: SWIR-Cirrus (1375nm, 60m)     - 권운 감지")
    print("  B11: SWIR (1610nm, 20m)            - 토양/식생 구분")
    print("  B12: SWIR (2190nm, 20m)            - 지질, 화재 감지")
    print()
    print("품질 밴드 (L2A 제품에만 포함):")
    print("  SCL: Scene Classification (20m)     - 픽셀 분류 맵")
    print("       값별 의미: 0=분류없음, 1=포화/결함, 2=짙은그림자, 3=구름그림자")
    print("                 4=식생, 5=비식생, 6=물, 7=구름(저신뢰), 8=구름(중신뢰)")
    print("                 9=구름(고신뢰), 10=얇은권운, 11=눈/얼음")
    print()
    print("💡 L1C vs L2A 제품:")
    print("  L1C: 대기상단 반사도 (TOA) - 대기 보정 전")
    print("  L2A: 지표면 반사도 (BOA) - 대기 보정 후, SCL 포함")
    print("  → 분석용도로는 L2A 제품 권장 (SCL로 구름/그림자 마스킹 가능)")
    print()
    print("⚡ 성능 팁:")
    print("===========")
    print("• 대용량 처리시 target_resolution=20 사용으로 속도 향상")
    print("• batch 모드에서 skip_existing=True로 중복 처리 방지")
    print("• SCL 밴드로 구름 픽셀 제외하여 분석 품질 향상")
    print("• 메모리 부족시 밴드 수를 줄여서 처리")
    print()
    print("🚨 주의사항:")
    print("============")
    print("• SCL 밴드는 L2A 제품에서만 사용 가능")
    print("• L1C 제품에서 SCL 모드 사용시 오류 발생")
    print("• 10m 해상도로 리샘플링시 파일 크기 증가")
    print("• ZIP 파일은 자동으로 압축 해제됨")

# 명령행에서 실행할 경우
if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_usage()
        print("\n사용 예제:")
        print(f"python {sys.argv[0]} /path/to/sentinel2_file.zip basic")
        print(f"python {sys.argv[0]} /path/to/sentinel2_file.zip vegetation")
        print(f"python {sys.argv[0]} /path/to/sentinel2_file.zip hand B02,B03,B04,B08")
        print(f"python {sys.argv[0]} /path/to/directory batch vegetation")
    else:
        input_path = sys.argv[1]

        # 디렉터리인지 파일인지 확인
        is_directory = os.path.isdir(input_path)
        is_batch_mode = len(sys.argv) > 2 and sys.argv[2].lower() == "batch"

        try:
            if is_directory or is_batch_mode:
                # 디렉터리 일괄 변환 모드
                if not is_directory:
                    print("❌ 오류: batch 모드에서는 디렉터리 경로를 입력해야 합니다.")
                    sys.exit(1)

                # 배치 모드일 때 모드는 3번째 인자
                mode = sys.argv[3].lower() if len(sys.argv) > 3 else "vegetation"
                custom_bands = None

                # target_resolution 파라미터 확인
                target_resolution = 10  # 기본값
                for arg in sys.argv:
                    if arg.startswith('target_resolution='):
                        try:
                            target_resolution = int(arg.split('=')[1])
                            if target_resolution not in [10, 20, 60]:
                                print(f"❌ 오류: target_resolution은 10, 20, 60 중 하나여야 합니다. 입력값: {target_resolution}")
                                sys.exit(1)
                        except ValueError:
                            print(f"❌ 오류: target_resolution 값이 올바르지 않습니다: {arg}")
                            sys.exit(1)

                if mode == "hand":
                    if len(sys.argv) < 5:
                        print("❌ 오류: batch hand 모드에서는 밴드를 지정해야 합니다.")
                        print("사용법: python jp2_tif_converter.py 디렉터리 batch hand B02,B03,B04,SCL")
                        print("       python jp2_tif_converter.py 디렉터리 batch hand B02,B03,B04,SCL target_resolution=20")
                        sys.exit(1)

                    bands_input = sys.argv[4]
                    custom_bands = [band.strip().upper() for band in bands_input.split(',')]

                    # 밴드 유효성 검사
                    valid_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'SCL']
                    invalid_bands = [b for b in custom_bands if b not in valid_bands]

                    if invalid_bands:
                        print(f"❌ 오류: 잘못된 밴드가 포함되어 있습니다: {', '.join(invalid_bands)}")
                        print(f"사용 가능한 밴드: {', '.join(valid_bands)}")
                        sys.exit(1)

                # 일괄 변환 실행
                results = batch_convert_sentinel2(input_path, mode, custom_bands, target_resolution=target_resolution)

                # 최종 결과 출력
                if results["total"] > 0:
                    success_rate = (results["success"] / results["total"]) * 100
                    print(f"\n🎉 일괄 변환 완료! 성공률: {success_rate:.1f}%")
                else:
                    print("\n❌ 변환할 파일이 없습니다.")

            else:
                # 단일 파일 변환 모드 (기존 코드)
                safe_path = input_path
                mode = sys.argv[2].lower() if len(sys.argv) > 2 else "basic"

                if mode == "basic":
                    print("=== 기본 모드: RGB + NIR 밴드 ===")
                    result = convert_rgb_nir(safe_path)

                elif mode == "vegetation":
                    print("=== 식생 분석 모드: 모든 식생 관련 밴드 ===")
                    result = convert_vegetation_bands(safe_path)

                elif mode == "scl":
                    print("=== SCL 모드: RGB+NIR+SCL 밴드 (L2A 전용) ===")
                    result = convert_with_scl(safe_path, include_all_bands=False)

                elif mode == "classification":
                    print("=== 분류 모드: 모든 스펙트럼 밴드 + SCL ===")
                    result = convert_classification_bands(safe_path)

                elif mode == "scl_only":
                    print("=== SCL 전용 모드: SCL 밴드만 (품질 평가용) ===")
                    result = convert_only_scl(safe_path)

                elif mode == "hand":
                    if len(sys.argv) < 4:
                        print("❌ 오류: hand 모드에서는 밴드를 지정해야 합니다.")
                        print("사용법: python jp2_tif_converter.py 파일.zip hand B02,B03,B04,SCL")
                        print("       python jp2_tif_converter.py 파일.zip hand B02,B03,B04,SCL target_resolution=20")
                        print("\n사용 가능한 밴드:")
                        print("B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12,SCL")
                        sys.exit(1)

                    bands_input = sys.argv[3]
                    bands = [band.strip().upper() for band in bands_input.split(',')]

                    # target_resolution 파라미터 확인
                    target_resolution = 10  # 기본값
                    for arg in sys.argv[4:]:
                        if arg.startswith('target_resolution='):
                            try:
                                target_resolution = int(arg.split('=')[1])
                                if target_resolution not in [10, 20, 60]:
                                    print(f"❌ 오류: target_resolution은 10, 20, 60 중 하나여야 합니다. 입력값: {target_resolution}")
                                    sys.exit(1)
                            except ValueError:
                                print(f"❌ 오류: target_resolution 값이 올바르지 않습니다: {arg}")
                                sys.exit(1)

                    print(f"=== 수동 모드: 사용자 지정 밴드 ===")
                    print(f"지정된 밴드: {', '.join(bands)}")
                    print(f"목표 해상도: {target_resolution}m")

                    # 밴드 유효성 검사
                    valid_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'SCL']
                    invalid_bands = [b for b in bands if b not in valid_bands]

                    if invalid_bands:
                        print(f"❌ 오류: 잘못된 밴드가 포함되어 있습니다: {', '.join(invalid_bands)}")
                        print(f"사용 가능한 밴드: {', '.join(valid_bands)}")
                        sys.exit(1)

                    result = convert_sentinel2_to_tiff(safe_path, bands=bands, target_resolution=target_resolution)

                else:
                    print(f"❌ 오류: 알 수 없는 모드 '{mode}'")
                    print("사용 가능한 모드: basic, vegetation, scl, classification, scl_only, hand")
                    print("\n사용 예제:")
                    print(f"python {sys.argv[0]} 파일.zip basic")
                    print(f"python {sys.argv[0]} 파일.zip vegetation")
                    print(f"python {sys.argv[0]} 파일.zip scl")
                    print(f"python {sys.argv[0]} 파일.zip classification")
                    print(f"python {sys.argv[0]} 파일.zip scl_only")
                    print(f"python {sys.argv[0]} 파일.zip hand B02,B03,B04,SCL")
                    sys.exit(1)

                print(f"\n✅ 성공적으로 변환되었습니다: {result}")

        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            sys.exit(1)