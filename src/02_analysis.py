"""
02_analysis.py  ─  통계 분석 (논문 표 재현)

분석 목록:
  1. 기술통계 (전체 / 주거형태별)
  2. 독립표본 t-검정 (Welch) : 주거형태 × 만족도
  3. 교차분석 + Fisher 정확검정 : 주거형태 × 만족도그룹
  4. 교차분석 + Fisher 정확검정 : 주거형태 × 정책인지도
  5. Spearman 순위상관 : 주거비용 × 만족도
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT    = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')

SEP = "─" * 60


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['만족도그룹'] = pd.Categorical(
        df['만족도그룹'],
        categories=['불만족(1-2)', '보통(3)', '만족(4-5)'],
        ordered=True,
    )
    return df


# ── 1. 기술통계 ───────────────────────────────────────────────────────────────
def descriptive(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  1. 기술통계")
    print(SEP)

    print("\n[주거 만족도 전체]")
    s = df['주거만족도']
    print(f"  N={len(s)}, M={s.mean():.2f}, SD={s.std(ddof=1):.2f}, "
          f"Min={s.min()}, Max={s.max()}")

    print("\n[주거형태별 만족도]")
    g = df.groupby('주거형태')['주거만족도']
    print(g.agg(['count', 'mean', 'std']).rename(
        columns={'count': 'N', 'mean': 'M', 'std': 'SD'}).round(2))

    print("\n[만족도그룹 분포 (주거형태별 %)]")
    ct = pd.crosstab(df['주거형태'], df['만족도그룹'], normalize='index') * 100
    print(ct.round(1))

    print("\n[주거지 탐색 방법 분포]")
    sr = df['주거지탐색방법'].value_counts()
    pct = (sr / len(df) * 100).round(1)
    print(pd.DataFrame({'빈도': sr, '%': pct}))

    print("\n[정책 인지도]")
    pi = df['정책인지도'].value_counts()
    print(pd.DataFrame({'빈도': pi, '%': (pi / len(df) * 100).round(1)}))

    print("\n[국적 분포 (상위 10개국)]")
    nc = df['국적'].value_counts().head(10)
    print(pd.DataFrame({'빈도': nc, '%': (nc / len(df) * 100).round(1)}))


# ── 2. Welch t-검정 ──────────────────────────────────────────────────────────
def ttest_satisfaction(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  2. 독립표본 t-검정 (Welch) : 주거형태 × 주거만족도")
    print(SEP)

    dorm = df.loc[df['주거형태'] == '기숙사',     '주거만족도']
    non  = df.loc[df['주거형태'] == '기숙사 외',  '주거만족도']

    t, p = stats.ttest_ind(non, dorm, equal_var=False)

    print(f"  기숙사    : N={len(dorm)}, M={dorm.mean():.2f}, SD={dorm.std(ddof=1):.2f}")
    print(f"  기숙사 외 : N={len(non)},  M={non.mean():.2f}, SD={non.std(ddof=1):.2f}")
    print(f"\n  t = {t:.3f},  p = {p:.3f}  {'*' if p < .05 else '(n.s.)'}")
    print("  → 기숙사 외 거주자의 만족도가 유의하게 높음" if p < .05 else
          "  → 유의미한 차이 없음")


# ── 3. 교차분석 : 주거형태 × 만족도그룹 ─────────────────────────────────────
def chi2_satisfaction_group(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  3. 교차분석 : 주거형태 × 만족도그룹")
    print(SEP)

    ct = pd.crosstab(df['주거형태'], df['만족도그룹'])
    print(ct)

    _, p_fisher = stats.fisher_exact(
        [[ct.loc['기숙사', '불만족(1-2)'],    ct.loc['기숙사', '만족(4-5)']],
         [ct.loc['기숙사 외', '불만족(1-2)'], ct.loc['기숙사 외', '만족(4-5)']]]
    )
    chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)

    print(f"\n  χ²({dof}) = {chi2:.3f},  p = {p_chi2:.3f}")
    print(f"  Fisher 정확검정 p = {p_fisher:.3f}  {'*' if p_fisher < .05 else '(n.s.)'}")


# ── 4. 교차분석 : 주거형태 × 정책 인지도 ────────────────────────────────────
def chi2_policy(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  4. 교차분석 : 주거형태 × 정책 인지도")
    print(SEP)

    ct = pd.crosstab(df['주거형태'], df['정책인지도'])
    print(ct)

    _, p_fisher = stats.fisher_exact(ct.values)
    chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)

    print(f"\n  χ²({dof}) = {chi2:.3f},  p = {p_chi2:.3f}")
    print(f"  Fisher 정확검정 p = {p_fisher:.3f}  {'*' if p_fisher < .05 else '(n.s.)'}")

    dorm_aware = ct.loc['기숙사', '예'] / ct.loc['기숙사'].sum() * 100
    non_aware  = ct.loc['기숙사 외', '예'] / ct.loc['기숙사 외'].sum() * 100
    print(f"\n  정책 인지율  기숙사: {dorm_aware:.1f}%  / 기숙사 외: {non_aware:.1f}%")


# ── 5. Spearman 상관 : 비용 × 만족도 ────────────────────────────────────────
def spearman_cost_sat(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  5. Spearman 순위상관 : 월평균주거비용 × 주거만족도")
    print(SEP)

    rho, p = stats.spearmanr(df['비용_ord'].dropna(), df['주거만족도'])
    print(f"  ρ = {rho:.3f},  p = {p:.3f}  {'*' if p < .05 else '(n.s.)'}")
    print("  → 비용과 만족도 사이에 유의미한 상관 없음" if p >= .05 else
          "  → 유의미한 상관 있음")


def main():
    df = load(INPUT)

    descriptive(df)
    ttest_satisfaction(df)
    chi2_satisfaction_group(df)
    chi2_policy(df)
    spearman_cost_sat(df)

    print(f"\n{SEP}")
    print("  분석 완료")
    print(SEP)


if __name__ == '__main__':
    main()
