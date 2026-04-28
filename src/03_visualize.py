"""
03_visualize.py  ─  시각화 (7개 Figure → outputs/figures/)

Figure 목록:
  01_satisfaction_by_housing.png   주거형태별 만족도 분포 (누적 막대)
  02_satisfaction_boxplot.png      만족도 박스플롯 비교
  03_search_method.png             주거 탐색 방법 (수평 막대)
  04_cost_vs_satisfaction.png      비용 × 만족도 산점도 (Spearman)
  05_policy_awareness.png          정책 인지도 도넛 차트
  06_nationality.png               국적 분포 (상위 10개)
  07_cost_distribution.png         주거형태별 비용 분포
"""

import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT    = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
FIG_DIR  = os.path.join(BASE_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE  = {'기숙사': '#E07B54', '기숙사 외': '#4C8DBF'}
DPI      = 150


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['만족도그룹'] = pd.Categorical(
        df['만족도그룹'],
        categories=['불만족(1-2)', '보통(3)', '만족(4-5)'],
        ordered=True,
    )
    return df


def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장 → {name}")


# ── Fig 01 : 주거형태별 만족도그룹 누적 막대 ─────────────────────────────────
def fig01_stacked_satisfaction(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df['주거형태'], df['만족도그룹'], normalize='index') * 100
    ct = ct[['불만족(1-2)', '보통(3)', '만족(4-5)']]

    colors = ['#D9534F', '#F0AD4E', '#5CB85C']
    fig, ax = plt.subplots(figsize=(7, 4))
    ct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white', width=0.5)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center',
                     fontsize=9, color='white', fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('비율 (%)')
    ax.set_title('주거형태별 만족도 분포', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticklabels(ct.index, rotation=0)
    ax.legend(title='만족도 그룹', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0, 110)
    ax.spines[['top', 'right']].set_visible(False)

    save(fig, '01_satisfaction_by_housing.png')


# ── Fig 02 : 만족도 박스플롯 + 개별 점 ───────────────────────────────────────
def fig02_boxplot(df: pd.DataFrame) -> None:
    dorm = df.loc[df['주거형태'] == '기숙사',    '주거만족도']
    non  = df.loc[df['주거형태'] == '기숙사 외', '주거만족도']
    _, p = stats.ttest_ind(non, dorm, equal_var=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x='주거형태', y='주거만족도', hue='주거형태',
                palette=PALETTE, width=0.45, linewidth=1.5, fliersize=0,
                legend=False, ax=ax)
    sns.stripplot(data=df, x='주거형태', y='주거만족도', hue='주거형태',
                  palette=PALETTE, alpha=0.45, jitter=0.12, size=5,
                  legend=False, ax=ax)

    for i, (grp, m) in enumerate(
            df.groupby('주거형태')['주거만족도'].mean().items()):
        ax.text(i, m + 0.12, f'M={m:.2f}', ha='center', fontsize=10,
                color='black', fontweight='bold')

    sig_txt = f'p = {p:.3f}{"*" if p < .05 else " (n.s.)"}'
    y_max = df['주거만족도'].max() + 0.6
    ax.plot([0, 0, 1, 1], [y_max - 0.3, y_max, y_max, y_max - 0.3],
            lw=1.2, c='gray')
    ax.text(0.5, y_max + 0.05, sig_txt, ha='center', fontsize=10, color='gray')

    ax.set_xlabel('')
    ax.set_ylabel('주거 만족도 (5점 척도)')
    ax.set_title('주거형태별 만족도 비교 (Welch t-test)', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_ylim(0.5, y_max + 0.5)
    ax.spines[['top', 'right']].set_visible(False)

    save(fig, '02_satisfaction_boxplot.png')


# ── Fig 03 : 주거 탐색 방법 수평 막대 ───────────────────────────────────────
def fig03_search_method(df: pd.DataFrame) -> None:
    order = ['학교·기숙사 배정', '지인 추천', '온라인 플랫폼', '부동산 중개', '기타']
    sr = df['주거지탐색방법'].value_counts().reindex(order)
    pct = sr / len(df) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#4C8DBF', '#6BAED6', '#9ECAE1', '#C6DBEF', '#DEEBF7']
    bars = ax.barh(order[::-1], pct[order[::-1]], color=colors[::-1],
                   edgecolor='white', height=0.55)

    for bar, p in zip(bars, pct[order[::-1]]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{p:.1f}%', va='center', fontsize=10)

    ax.set_xlabel('응답 비율 (%)')
    ax.set_title('주거지 탐색 방법 분포 (N=69)', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_xlim(0, 40)
    ax.spines[['top', 'right']].set_visible(False)

    save(fig, '03_search_method.png')


# ── Fig 04 : 비용 × 만족도 산점도 (Spearman) ────────────────────────────────
def fig04_cost_vs_satisfaction(df: pd.DataFrame) -> None:
    x = df['비용_ord'].dropna()
    y = df.loc[x.index, '주거만족도']
    rho, p = stats.spearmanr(x, y)

    jitter_x = x + np.random.uniform(-0.15, 0.15, len(x))
    jitter_y = y + np.random.uniform(-0.12, 0.12, len(y))

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(jitter_x, jitter_y,
                         c=df.loc[x.index, '기숙사여부'],
                         cmap='coolwarm', alpha=0.65, s=55, edgecolors='w')

    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(1, 5, 100)
    ax.plot(xline, m * xline + b, '--', color='gray', linewidth=1.4)

    labels = ['20만원 미만', '20-30만원', '30-40만원', '40-50만원', '50만원 이상']
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(1, 6))
    ax.set_xlabel('월 평균 주거비용')
    ax.set_ylabel('주거 만족도')
    ax.set_title(f'주거비용 × 만족도 (Spearman ρ = {rho:.3f}, p = {p:.3f})',
                 fontsize=12, fontweight='bold', pad=12)

    patches = [mpatches.Patch(color='#B22222', label='기숙사'),
               mpatches.Patch(color='#4169E1', label='기숙사 외')]
    ax.legend(handles=patches, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    save(fig, '04_cost_vs_satisfaction.png')


# ── Fig 05 : 정책 인지도 도넛 ────────────────────────────────────────────────
def fig05_policy_donut(df: pd.DataFrame) -> None:
    vc = df['정책인지도'].value_counts()
    yes = vc.get('예', 0)
    no  = vc.get('아니요', 0)

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        [yes, no],
        labels=[f'인지\n({yes}명)', f'미인지\n({no}명)'],
        autopct='%1.1f%%',
        colors=['#E07B54', '#D0D0D0'],
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor='white'),
        textprops={'fontsize': 11},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')

    ax.set_title('주거 지원 정책 인지도 (N=69)', fontsize=13,
                 fontweight='bold', pad=14)
    save(fig, '05_policy_awareness.png')


# ── Fig 06 : 국적 분포 (상위 10개) ──────────────────────────────────────────
def fig06_nationality(df: pd.DataFrame) -> None:
    nc = df['국적'].value_counts().head(10)
    colors = sns.color_palette('Blues_r', len(nc))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(nc.index, nc.values, color=colors, edgecolor='white')

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(int(bar.get_height())),
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('')
    ax.set_ylabel('인원 수 (명)')
    ax.set_title('응답자 국적 분포 (상위 10개국)', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_xticks(range(len(nc)))
    ax.set_xticklabels(nc.index, rotation=30, ha='right')
    ax.spines[['top', 'right']].set_visible(False)

    save(fig, '06_nationality.png')


# ── Fig 07 : 주거형태별 비용 분포 ───────────────────────────────────────────
def fig07_cost_distribution(df: pd.DataFrame) -> None:
    order = ['20만원 미만', '20-30만원', '30-40만원', '40-50만원', '50만원 이상']
    ct = pd.crosstab(df['월평균주거비용'], df['주거형태']).reindex(order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ct.plot(kind='bar', ax=ax, color=[PALETTE['기숙사'], PALETTE['기숙사 외']],
            edgecolor='white', width=0.6)

    ax.set_xlabel('')
    ax.set_ylabel('인원 수 (명)')
    ax.set_title('주거형태별 월평균 주거비용 분포', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_xticklabels(order, rotation=20, ha='right')
    ax.legend(title='주거형태')
    ax.spines[['top', 'right']].set_visible(False)

    save(fig, '07_cost_distribution.png')


def main():
    df = load(INPUT)
    print("시각화 생성 중...")

    fig01_stacked_satisfaction(df)
    fig02_boxplot(df)
    fig03_search_method(df)
    fig04_cost_vs_satisfaction(df)
    fig05_policy_donut(df)
    fig06_nationality(df)
    fig07_cost_distribution(df)

    print(f"\n[완료] 7개 Figure 저장 → {FIG_DIR}")


if __name__ == '__main__':
    main()
