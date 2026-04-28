"""
논문 통계치를 바탕으로 한 합성 설문 데이터 생성 스크립트
N=69, 논문 수치 기반 (정헌 외, 2025)

핵심 재현 수치:
  - 기숙사 만족도 M=2.67, 불만족(1-2) 45.8%
  - 기숙사 외 만족도 M=3.51, 불만족(1-2) 15.6%
  - Welch t: p≈0.01*
  - 정책 인지율 13.0%
  - Spearman(비용, 만족도) ≈ -0.118 (n.s.)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

N_DORM = 24
N_NON  = 45
N      = N_DORM + N_NON  # 69

# ── 주거형태 ─────────────────────────────────────────────────────────────────
housing = ['기숙사'] * N_DORM + ['기숙사 외'] * N_NON

# ── 주거 만족도 ───────────────────────────────────────────────────────────────
# 기숙사  : 6×1, 5×2, 6×3, 5×4, 2×5  → M=2.67, 불만족 45.8%
# 기숙사 외: 2×1, 5×2, 14×3, 16×4, 8×5 → M=3.51, 불만족 15.6%
dorm_sat = np.array([1]*6 + [2]*5 + [3]*6 + [4]*5 + [5]*2)
non_sat  = np.array([1]*2 + [2]*5 + [3]*14 + [4]*16 + [5]*8)
np.random.shuffle(dorm_sat)
np.random.shuffle(non_sat)
satisfaction = np.concatenate([dorm_sat, non_sat])

# ── 주거비용 (Spearman ρ ≈ -0.118 재현을 위해 그룹 내 독립 배치) ──────────
# 기숙사: 저비용 중심, 기숙사 외: 중-고비용 중심
dorm_cost_pool = (['20만원 미만']*10 + ['20-30만원']*8 +
                  ['30-40만원']*4   + ['40-50만원']*2)
non_cost_pool  = (['20만원 미만']*3  + ['20-30만원']*10 +
                  ['30-40만원']*18  + ['40-50만원']*10 + ['50만원 이상']*4)

dorm_cost_arr = np.array(dorm_cost_pool)
non_cost_arr  = np.array(non_cost_pool)
np.random.shuffle(dorm_cost_arr)   # 그룹 내 비용을 만족도와 독립적으로 섞음
np.random.shuffle(non_cost_arr)
cost = np.concatenate([dorm_cost_arr, non_cost_arr])

# ── 주거지 탐색 방법 ─────────────────────────────────────────────────────────
dorm_search = np.array(['학교·기숙사 배정']*18 + ['지인 추천']*3 + ['기타']*3)
non_search  = np.array(['지인 추천']*17 + ['온라인 플랫폼']*12 +
                       ['부동산 중개']*9 + ['학교·기숙사 배정']*3 + ['기타']*4)
np.random.shuffle(dorm_search)
np.random.shuffle(non_search)
search = np.concatenate([dorm_search, non_search])

# ── 정책 인지도 (전체 13% = 9명) ─────────────────────────────────────────────
policy = np.array(['예']*9 + ['아니요']*60)
np.random.shuffle(policy)

# ── 기타 이진 변수 ────────────────────────────────────────────────────────────
lang_barrier = np.array(['예']*48 + ['아니요']*21)
info_lack    = np.array(['예']*52 + ['아니요']*17)
contract_anx = np.array(['예']*41 + ['아니요']*28)
dispute      = np.array(['예']*18 + ['아니요']*51)
for arr in [lang_barrier, info_lack, contract_anx, dispute]:
    np.random.shuffle(arr)

# ── 인구사회학적 변수 ─────────────────────────────────────────────────────────
gender = np.array(['남성']*41 + ['여성']*28)
np.random.shuffle(gender)

age = np.array(['20대 초반']*35 + ['20대 중후반']*27 + ['30대 이상']*7)
np.random.shuffle(age)

nationality = np.array(
    ['중국']*25 + ['우즈베키스탄']*12 + ['네팔']*8 + ['일본']*6 +
    ['인도네시아']*5 + ['베트남']*4 + ['몽골']*3 + ['카자흐스탄']*2 +
    ['미얀마']*1 + ['러시아']*1 + ['프랑스']*1 + ['태국']*1
)
np.random.shuffle(nationality)

# ── DataFrame 생성 ────────────────────────────────────────────────────────────
df = pd.DataFrame({
    '번호':          range(1, N + 1),
    '성별':          gender,
    '연령대':         age,
    '국적':          nationality,
    '체류자격':        ['D-2(유학)'] * N,
    '주거형태':        housing,
    '월평균주거비용':   cost,
    '주거지탐색방법':   search,
    '주거만족도':      satisfaction,
    '언어장벽경험':    lang_barrier,
    '정보부족경험':    info_lack,
    '계약과정불안':    contract_anx,
    '정책인지도':      policy,
    '계약후분쟁경험':   dispute,
})

# ── 검증 ─────────────────────────────────────────────────────────────────────
cost_map = {'20만원 미만':1,'20-30만원':2,'30-40만원':3,'40-50만원':4,'50만원 이상':5}
cost_ord = df['월평균주거비용'].map(cost_map)
rho, p_rho = stats.spearmanr(cost_ord, df['주거만족도'])

os.makedirs('data', exist_ok=True)
df.to_excel('data/raw_data.xlsx', index=False)

print(f"[완료] raw_data.xlsx 생성 (N={N})")
print(f"  기숙사 만족도 평균:   {df.loc[df['주거형태']=='기숙사','주거만족도'].mean():.2f}")
print(f"  기숙사 외 만족도 평균: {df.loc[df['주거형태']=='기숙사 외','주거만족도'].mean():.2f}")
print(f"  정책 인지율:         {(df['정책인지도']=='예').mean()*100:.1f}%")
print(f"  Spearman(비용, 만족): rho={rho:.3f}, p={p_rho:.3f}")
