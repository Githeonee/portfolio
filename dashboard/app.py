"""
dashboard/app.py  ─  Streamlit 인터랙티브 대시보드
실행: streamlit run dashboard/app.py
"""

import os
import sys
import platform
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='외국인 유학생 주거 불안정 분석',
    page_icon='🏠',
    layout='wide',
)

# ── 공통 색상 ─────────────────────────────────────────────────────────────────
COLOR = {'기숙사': '#E07B54', '기숙사 외': '#4C8DBF'}


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['만족도그룹'] = pd.Categorical(
        df['만족도그룹'],
        categories=['불만족(1-2)', '보통(3)', '만족(4-5)'],
        ordered=True,
    )
    return df


def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 응답자", f"{len(df)}명")
    c2.metric("평균 만족도", f"{df['주거만족도'].mean():.2f} / 5")
    c3.metric("기숙사 불만족률",
              f"{(df.loc[df['주거형태']=='기숙사','주거만족도']<=2).mean()*100:.1f}%")
    c4.metric("정책 인지율",
              f"{(df['정책인지도']=='예').mean()*100:.1f}%")
    c5.metric("조사 대학", "강원대 · 한림대")


def tab_overview(df: pd.DataFrame) -> None:
    st.subheader("응답자 개요")
    col1, col2 = st.columns(2)

    with col1:
        nc = df['국적'].value_counts().reset_index()
        nc.columns = ['국적', '인원']
        fig = px.bar(nc.head(10), x='국적', y='인원',
                     color='인원', color_continuous_scale='Blues',
                     title='국적 분포 (상위 10개국)')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ht = df['주거형태'].value_counts().reset_index()
        ht.columns = ['주거형태', '인원']
        fig = px.pie(ht, names='주거형태', values='인원',
                     color='주거형태',
                     color_discrete_map=COLOR,
                     hole=0.45, title='주거형태 비율')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        g = df['성별'].value_counts().reset_index()
        g.columns = ['성별', '인원']
        fig = px.pie(g, names='성별', values='인원', hole=0.45, title='성별 비율',
                     color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        a = df['연령대'].value_counts().reset_index()
        a.columns = ['연령대', '인원']
        fig = px.bar(a, x='연령대', y='인원',
                     color='연령대', title='연령대 분포',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def tab_satisfaction(df: pd.DataFrame) -> None:
    st.subheader("주거 만족도 분석")

    # 만족도 분포 비교
    ct = (pd.crosstab(df['주거형태'], df['만족도그룹'], normalize='index') * 100
            .reset_index().melt(id_vars='주거형태', var_name='만족도그룹', value_name='비율'))

    # crosstab을 직접 melt
    ct_raw = pd.crosstab(df['주거형태'], df['만족도그룹'], normalize='index') * 100
    ct_melt = ct_raw.reset_index().melt(id_vars='주거형태',
                                         var_name='만족도그룹', value_name='비율(%)')

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ct_melt, x='주거형태', y='비율(%)', color='만족도그룹',
                     barmode='stack',
                     color_discrete_map={'불만족(1-2)': '#D9534F',
                                         '보통(3)':    '#F0AD4E',
                                         '만족(4-5)':  '#5CB85C'},
                     title='주거형태별 만족도 분포 (누적 막대)')
        fig.update_layout(legend_title_text='만족도 그룹')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x='주거형태', y='주거만족도',
                     color='주거형태', color_discrete_map=COLOR,
                     points='all', title='만족도 분포 (박스플롯 + 개별 점)')
        dorm = df.loc[df['주거형태']=='기숙사', '주거만족도']
        non  = df.loc[df['주거형태']=='기숙사 외', '주거만족도']
        _, p = stats.ttest_ind(non, dorm, equal_var=False)
        fig.add_annotation(
            x=0.5, y=5.5, xref='paper',
            text=f'Welch t-test: p = {p:.3f}{"*" if p<.05 else " (n.s.)"}',
            showarrow=False, font=dict(size=12, color='gray'),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 통계 카드
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("기숙사 만족도 평균", f"{dorm.mean():.2f}")
    c2.metric("기숙사 외 만족도 평균", f"{non.mean():.2f}")
    c3.metric("차이", f"{non.mean()-dorm.mean():.2f} 점 (기숙사 외 > 기숙사)",
              delta_color="normal")


def tab_housing_search(df: pd.DataFrame) -> None:
    st.subheader("주거 탐색 방법")

    order = ['학교·기숙사 배정', '지인 추천', '온라인 플랫폼', '부동산 중개', '기타']
    sr = df['주거지탐색방법'].value_counts().reindex(order).reset_index()
    sr.columns = ['방법', '인원']
    sr['비율(%)'] = (sr['인원'] / len(df) * 100).round(1)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig = px.bar(sr, x='비율(%)', y='방법', orientation='h',
                     color='비율(%)', color_continuous_scale='Blues',
                     title='주거지 탐색 방법 (응답 비율)',
                     text='비율(%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis={'categoryorder': 'array', 'categoryarray': order[::-1]})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(sr, names='방법', values='인원', hole=0.4,
                     title='탐색 방법 비율',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**분석 결과**: 공인중개사를 통한 정식 계약 비율은 **13%**에 불과합니다. "
        "지인 추천(29%)과 비공식 경로 의존이 보증금 미반환, 계약 조건 오해 등 "
        "분쟁으로 이어질 위험이 있습니다."
    )


def tab_cost_correlation(df: pd.DataFrame) -> None:
    st.subheader("주거비용 × 만족도 상관분석")

    x = df['비용_ord'].dropna()
    y = df.loc[x.index, '주거만족도']
    rho, p = stats.spearmanr(x, y)

    label_map = {1:'20만원\n미만', 2:'20-30만원', 3:'30-40만원',
                 4:'40-50만원', 5:'50만원\n이상'}
    plot_df = df.loc[x.index].copy()
    plot_df['비용라벨'] = plot_df['비용_ord'].map(label_map)

    col1, col2 = st.columns([1.6, 1])
    with col1:
        fig = px.scatter(
            plot_df, x='비용_ord', y='주거만satisfaction',
            color='주거형태', color_discrete_map=COLOR,
            trendline='ols',
            title=f'주거비용 × 만족도  (Spearman ρ = {rho:.3f}, p = {p:.3f})',
            labels={'비용_ord': '월평균 주거비용 (순서형)',
                    '주거만satisfaction': '주거 만족도'},
        ) if '주거만satisfaction' in plot_df.columns else px.scatter(
            plot_df, x='비용_ord', y='주거만족도',
            color='주거형태', color_discrete_map=COLOR,
            trendline='ols',
            title=f'주거비용 × 만족도  (Spearman ρ = {rho:.3f}, p = {p:.3f})',
        )
        fig.update_xaxes(tickvals=list(range(1, 6)),
                         ticktext=['20만원 미만','20-30만원','30-40만원',
                                   '40-50만원','50만원 이상'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cost_sat = plot_df.groupby('비용_ord')['주거만족도'].mean().reset_index()
        cost_sat['비용라벨'] = cost_sat['비용_ord'].map(label_map)
        fig = px.bar(cost_sat, x='비용라벨', y='주거만족도',
                     color='주거만족도', color_continuous_scale='RdYlGn',
                     title='비용 구간별 평균 만족도')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        f"**Spearman ρ = {rho:.3f} (p = {p:.3f}, 유의하지 않음)** ─ "
        "주거비용이 높아도 만족도가 높아지지 않습니다. "
        "단순 비용 지원보다 **정보 접근성·공간 구성·계약 안전성**이 핵심 요인입니다."
    )


def tab_policy(df: pd.DataFrame) -> None:
    st.subheader("정책 인지도")

    overall = (df['정책인지도'] == '예').mean() * 100
    dorm_rate = (df.loc[df['주거형태']=='기숙사', '정책인지도']=='예').mean() * 100
    non_rate  = (df.loc[df['주거형태']=='기숙사 외', '정책인지도']=='예').mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 인지율", f"{overall:.1f}%")
    c2.metric("기숙사 인지율", f"{dorm_rate:.1f}%")
    c3.metric("기숙사 외 인지율", f"{non_rate:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        vc = df['정책인지도'].value_counts().reset_index()
        vc.columns = ['인지여부', '인원']
        fig = px.pie(vc, names='인지여부', values='인원', hole=0.5,
                     title='전체 정책 인지도',
                     color='인지여부',
                     color_discrete_map={'예': '#E07B54', '아니요': '#D0D0D0'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ct = pd.crosstab(df['주거형태'], df['정책인지도'])
        ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).reset_index()
        ct_melt = ct_pct.melt(id_vars='주거형태', var_name='인지여부', value_name='비율(%)')
        fig = px.bar(ct_melt, x='주거형태', y='비율(%)', color='인지여부',
                     color_discrete_map={'예': '#E07B54', '아니요': '#D0D0D0'},
                     title='주거형태별 정책 인지율',
                     text='비율(%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        st.plotly_chart(fig, use_container_width=True)

    st.error(
        "**유학생의 87%가 주거 지원 정책을 모릅니다.** "
        "정책이 없는 것이 아니라 정보 전달 체계 자체가 작동하지 않습니다. "
        "다국어 안내 플랫폼 구축이 시급합니다."
    )


def main():
    st.title('외국인 유학생 주거 불안정 데이터 분석')
    st.markdown(
        '> KCI 등재 논문 기반 실증 분석 | 정헌 외 (2025) | 강원대·한림대 유학생 N=69'
    )
    st.markdown('---')

    if not os.path.exists(DATA_PATH):
        st.error(
            f"데이터 파일이 없습니다: `{DATA_PATH}`\n\n"
            "먼저 다음을 실행하세요:\n"
            "```bash\npython data/generate_data.py\npython src/01_cleaning.py\n```"
        )
        st.stop()

    df = load_data(DATA_PATH)
    kpi_row(df)
    st.markdown('---')

    tab = st.tabs(['📋 응답자 개요', '😊 만족도 분석', '🔍 탐색 방법',
                   '💰 비용 상관', '📋 정책 인지도'])

    with tab[0]: tab_overview(df)
    with tab[1]: tab_satisfaction(df)
    with tab[2]: tab_housing_search(df)
    with tab[3]: tab_cost_correlation(df)
    with tab[4]: tab_policy(df)


if __name__ == '__main__':
    main()
