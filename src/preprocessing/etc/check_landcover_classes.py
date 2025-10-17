#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
토지피복 클래스 분포 확인
"""

import sqlite3
import pandas as pd

# 파일 경로
dir = r"D:\Landslide\labels\landcover_gyeongnam.gpkg"

print(f"파일: {dir}")
print("=" * 60)

try:
    # GPKG 파일을 SQLite로 연결
    conn = sqlite3.connect(dir)

    # 테이블 목록 확인
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("사용 가능한 테이블:")
    for table in tables['name']:
        print(f"  - {table}")

    # merge_landcover 테이블 사용 (데이터가 있는 테이블)
    if 'merge_landcover' in tables['name'].values:
        table_name = 'merge_landcover'
    else:
        table_name = tables['name'].iloc[0] if len(tables) > 0 else 'gpkg_contents'

    print(f"\n테이블 '{table_name}' 분석 중...")

    # 컬럼 정보 확인
    columns_info = pd.read_sql(f"PRAGMA table_info({table_name});", conn)
    print("\n컬럼 목록:")
    for _, col in columns_info.iterrows():
        print(f"  - {col['name']}: {col['type']}")

    # L2_NAME 컬럼이 있는지 확인
    if 'L2_NAME' in columns_info['name'].values:
        # 전체 데이터 개수
        total_count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name};", conn)['count'].iloc[0]
        print(f"\n전체 행 수: {total_count:,}")

        # L2_NAME 클래스 분포 확인
        query = f"SELECT L2_NAME, COUNT(*) as count FROM {table_name} GROUP BY L2_NAME ORDER BY count DESC;"
        class_counts = pd.read_sql(query, conn)

        print(f"\nL2_NAME 토지피복 클래스 분포:")
        print(f"총 {len(class_counts)}개 클래스")
        print("-" * 50)

        for _, row in class_counts.iterrows():
            class_name = row['L2_NAME']
            count = row['count']
            percentage = (count / total_count) * 100
            print(f"{class_name:20s}: {count:8,}개 ({percentage:5.1f}%)")
    else:
        print("L2_NAME 컬럼이 없습니다.")

    conn.close()

except Exception as e:
    print(f"오류: {e}")

    # 대안: 파일 구조 확인
    try:
        conn = sqlite3.connect(dir)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(f"테이블 목록: {cursor.fetchall()}")
        conn.close()
    except Exception as e2:
        print(f"테이블 확인도 실패: {e2}")