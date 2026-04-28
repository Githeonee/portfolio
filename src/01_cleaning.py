"""
01_cleaning.py  ─  데이터 전처리 파이프라인
raw_data.xlsx → cleaned_data.csv

처리 항목:
  - 열 이름 표준화
  - 결측치 확인 및 보고
  - 순서형 변수 수치 인코딩
  - 파생 변수 생성 (만족도그룹, 비용수치)
"""

import os
import pandas as pd
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT     = os.path.join(BASE_DIR, 'data', 'raw_data.xlsx')
OUTPUT    = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"[로드] {len(df)}행 × {len(df.columns)}열")
    return df


def check_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    if missing.any():
        print("[결측치 발견]")
        print(missing[missing > 0])
    else:
        print("[결측치] 없음")


def encode_cost(series: pd.Series) -> pd.Series:
    """월평균주거비용 → 순서형 정수 (1~5)"""
    mapping = {
        '20만원 미만':  1,
        '20-30만원':    2,
        '30-40만원':    3,
        '40-50만원':    4,
        '50만원 이상':  5,
    }
    return series.map(mapping).astype('Int64')


def encode_binary(series: pd.Series) -> pd.Series:
    """예/아니요 → 1/0"""
    return series.map({'예': 1, '아니요': 0}).astype('Int64')


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 만족도 그룹 (불만족=1-2, 보통=3, 만족=4-5)
    df['만족도그룹'] = pd.cut(
        df['주거만족도'],
        bins=[0, 2, 3, 5],
        labels=['불만족(1-2)', '보통(3)', '만족(4-5)'],
        right=True,
    )

    # 기숙사 거주 여부 이진 인코딩
    df['기숙사여부'] = (df['주거형태'] == '기숙사').astype(int)

    # 비용 중앙값 (만원 단위, 상관분석용)
    cost_mid = {1: 10, 2: 25, 3: 35, 4: 45, 5: 55}
    df['비용중앙값'] = df['비용_ord'].map(cost_mid)

    return df


def clean(path: str) -> pd.DataFrame:
    df = load_raw(path)
    check_missing(df)

    # 순서형 인코딩
    df['비용_ord'] = encode_cost(df['월평균주거비용'])

    # 이진 인코딩
    binary_cols = ['언어장벽경험', '정보부족경험', '계약과정불안', '정책인지도', '계약후분쟁경험']
    for col in binary_cols:
        df[f'{col}_bin'] = encode_binary(df[col])

    # 파생 변수
    df = add_derived_columns(df)

    # 컬럼 순서 정리
    df = df.sort_values('번호').reset_index(drop=True)

    print(f"[전처리 완료] 최종 {len(df)}행 × {len(df.columns)}열")
    print(df.dtypes.to_string())
    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df = clean(INPUT)
    df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 저장 → {OUTPUT}")
    print(df.head(3).to_string())


if __name__ == '__main__':
    main()
