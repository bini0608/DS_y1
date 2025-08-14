import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="매출 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Helpers
# ---------------------------
def embed_font_css(font_bytes: bytes, font_name: str = "NanumGothic") -> None:
    """
    업로드된 TTF를 base64로 인라인하여 웹폰트로 임베드.
    Plotly/Streamlit 텍스트에 동일 폰트명이 적용되도록 레이아웃 font.family를 이 이름으로 사용.
    """
    b64 = base64.b64encode(font_bytes).decode()
    css = f"""
    <style>
    @font-face {{
        font-family: '{font_name}';
        src: url(data:font/ttf;base64,{b64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}
    html, body, [class*="css"] {{
        font-family: '{font_name}', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif !important;
    }}
    .plotly text {{
        font-family: '{font_name}', 'Noto Sans KR', sans-serif !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def fig_layout_defaults(fig: go.Figure, title: str, font_family: str = "NanumGothic"):
    fig.update_layout(
        title=title,
        title_x=0.01,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        font=dict(family=font_family),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    return fig


def ensure_datetime_month(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.strftime("%Y-%m-%d")


# ---------------------------
# Sidebar: Inputs
# ---------------------------
st.sidebar.header("데이터 & 폰트 업로드")
excel_file = st.sidebar.file_uploader("엑셀 파일 업로드 (예: 0. 그래프_최종_과제용.xlsx)", type=["xlsx"])
font_file = st.sidebar.file_uploader("나눔고딕 TTF 업로드 (선택)", type=["ttf"])

font_family = "NanumGothic"
if font_file is not None:
    embed_font_css(font_file.read(), font_family)

st.sidebar.divider()
st.sidebar.caption("라벨 보이기 옵션")
show_all_point_labels = st.sidebar.checkbox("시계열 모든 지점 라벨 표시", value=False)
show_scatter_labels = st.sidebar.checkbox("산점도 좌표 라벨 표시", value=True)

st.sidebar.divider()
st.sidebar.caption("버블 겹침/표시 조정")
bubble_target_px = st.sidebar.slider("버블 최대 지름(픽셀)", 30, 120, 60, 5)
bubble_opacity = st.sidebar.slider("버블 투명도", 0.1, 1.0, 0.5, 0.05)
reduce_overlap = st.sidebar.checkbox("겹침 완화(지터 적용)", value=True)
jitter_strength = st.sidebar.slider("지터 강도", 0.0, 15.0, 6.0, 0.5)
show_bubble_labels = st.sidebar.checkbox("버블 라벨 표시", value=True)
label_positions = ["top center", "bottom center", "middle left", "middle right", "middle center"]
label_position_choice = st.sidebar.selectbox("라벨 위치", label_positions, index=0)

# ---------------------------
# Load data
# ---------------------------
if excel_file is None:
    st.info("왼쪽 사이드바에서 엑셀 파일을 업로드해주세요.")
    st.stop()

xls = pd.ExcelFile(excel_file)
# 시트 이름 그대로 사용 (문제의 엑셀 구조 기준)
bar_df = pd.read_excel(xls, sheet_name="바차트_히스토그램")
time_df = pd.read_excel(xls, sheet_name="시계열차트")
pie_df = pd.read_excel(xls, sheet_name="파이차트")
scatter_df = pd.read_excel(xls, sheet_name="산점도")
pareto_df = pd.read_excel(xls, sheet_name="파레토차트")
bubble_df = pd.read_excel(xls, sheet_name="버블차트")

# 공통 전처리
bar_df = bar_df.copy()
bar_df["월_str"] = ensure_datetime_month(bar_df["월"])

time_df = time_df.copy()
time_df["월_str"] = ensure_datetime_month(time_df["월"])

pie_df = pie_df.copy()
pie_labels_col = pie_df.columns[0]  # 보통 'Unnamed: 0'(제품명)
pie_val_col = "1분기 매출"

scatter_df = scatter_df.copy()
scatter_x = "비용"
scatter_y = "제품 A 매출"

pareto_df = pareto_df.copy().sort_values(by="매출", ascending=False)
pareto_df["누적비율(%)"] = pareto_df["매출"].cumsum() / pareto_df["매출"].sum() * 100

bubble_df = bubble_df.copy()
bubble_x = "마진"
bubble_y = "고객 수"
bubble_size = "제품별 비용"
bubble_text = "제품"

# ---------------------------
# Layout
# ---------------------------
st.title("📊 매출 분석 대시보드 (인터랙티브)")
st.caption("한글 폰트가 깨지면 사이드바에서 나눔고딕 TTF를 업로드해주세요.")

# ---- Row 1: 바차트 + 시계열
col1, col2 = st.columns(2, gap="large")

with col1:
    # 1) 바차트 (월별 총 매출)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_df["월_str"],
        y=bar_df["총 매출"],
        text=bar_df["총 매출"],
        textposition="outside",
        hovertemplate="월=%{x}<br>총 매출=%{y}<extra></extra>"
    ))
    fig_layout_defaults(fig_bar, "월별 총 매출", font_family)
    fig_bar.update_xaxes(title_text="월")
    fig_bar.update_yaxes(title_text="총 매출")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # 2) 시계열 (제품별 월 매출)
    fig_ts = go.Figure()
    for col in [c for c in time_df.columns if c not in ["월", "월_str"]]:
        fig_ts.add_trace(go.Scatter(
            x=time_df["월_str"],
            y=time_df[col],
            name=col,
            mode="lines+markers" + ("+text" if show_all_point_labels else ""),
            text=time_df[col] if show_all_point_labels else None,
            textposition="top center",
            hovertemplate="월=%{x}<br>"+col+"=%{y}<extra></extra>"
        ))
    fig_layout_defaults(fig_ts, "제품별 월 매출 추이", font_family)
    fig_ts.update_xaxes(title_text="월")
    fig_ts.update_yaxes(title_text="매출")
    st.plotly_chart(fig_ts, use_container_width=True)

# ---- Row 2: 파이 + 산점도(회귀선)
col3, col4 = st.columns(2, gap="large")

with col3:
    # 3) 파이차트 (1분기 매출 비중)
    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_df[pie_labels_col],
        values=pie_df[pie_val_col],
        hole=0.0,
        sort=False,
        textinfo="label+percent",
        hovertemplate="%{label}<br>매출=%{value} (%{percent})<extra></extra>"
    )])
    fig_layout_defaults(fig_pie, "제품별 1분기 매출 비중", font_family)
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    # 4) 산점도 + 회귀선 (제품 A 매출 vs 비용)
    X = scatter_df[[scatter_x]].values
    y = scatter_df[scatter_y].values
    model = LinearRegression().fit(X, y)
    x_min, x_max = float(np.min(X)), float(np.max(X))
    x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=scatter_df[scatter_x],
        y=scatter_df[scatter_y],
        mode="markers" + ("+text" if show_scatter_labels else ""),
        text=[f"({int(x)}, {int(v)})" for x, v in zip(scatter_df[scatter_x], scatter_df[scatter_y])] if show_scatter_labels else None,
        textposition="top center",
        name="데이터",
        hovertemplate=f"{scatter_x}=%{{x}}<br>{scatter_y}=%{{y}}<extra></extra>"
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_line.flatten(),
        y=y_line,
        mode="lines",
        name="회귀선",
        hovertemplate=f"예측 {scatter_y}=%{{y:.1f}}<extra></extra>"
    ))
    fig_layout_defaults(fig_scatter, "제품 A 매출 vs 비용", font_family)
    fig_scatter.update_xaxes(title_text=scatter_x)
    fig_scatter.update_yaxes(title_text=scatter_y)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---- Row 3: 파레토 + 버블
col5, col6 = st.columns(2, gap="large")

with col5:
    # 5) 파레토 차트 (막대 + 누적비율선)
    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(go.Bar(
        x=pareto_df["부서"],
        y=pareto_df["매출"],
        name="매출",
        text=pareto_df["매출"],
        textposition="outside",
        hovertemplate="부서=%{x}<br>매출=%{y}<extra></extra>"
    ), secondary_y=False)
    fig_pareto.add_trace(go.Scatter(
        x=pareto_df["부서"],
        y=pareto_df["누적비율(%)"],
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in pareto_df["누적비율(%)"]],
        textposition="top center",
        name="누적비율",
        hovertemplate="누적비율=%{y:.1f}%<extra></extra>"
    ), secondary_y=True)
    fig_layout_defaults(fig_pareto, "부서별 매출 파레토 분석", font_family)
    fig_pareto.update_xaxes(title_text="부서")
    fig_pareto.update_yaxes(title_text="매출", secondary_y=False)
    fig_pareto.update_yaxes(title_text="누적 비율 (%)", range=[0, 110], secondary_y=True)
    st.plotly_chart(fig_pareto, use_container_width=True)

with col6:
    # 6) 버블 차트 (비용·마진·고객 수) — 면적 스케일 + sizeref + 지터 + 투명도/라벨 옵션
    sizes_raw = bubble_df[bubble_size].to_numpy(dtype=float)

    # Plotly 권장 공식: sizeref = 2.*max(size)/(target_pixel**2)
    sizeref = 2.0 * sizes_raw.max() / (bubble_target_px ** 2)

    # 좌표 값 (지터 적용 옵션)
    if reduce_overlap:
        rng = np.random.default_rng(42)  # 재현 가능
        jx = rng.normal(0, jitter_strength, size=len(bubble_df))
        jy = rng.normal(0, jitter_strength, size=len(bubble_df))
        x_vals = bubble_df[bubble_x].to_numpy(dtype=float) + jx
        y_vals = bubble_df[bubble_y].to_numpy(dtype=float) + jy
    else:
        x_vals = bubble_df[bubble_x]
        y_vals = bubble_df[bubble_y]

    # 라벨 표시 여부/위치
    text_vals = bubble_df[bubble_text] if show_bubble_labels else None
    text_pos = label_position_choice if show_bubble_labels else "middle center"

    fig_bubble = go.Figure()
    fig_bubble.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers" + ("+text" if show_bubble_labels else ""),
        text=text_vals,
        textposition=text_pos,
        textfont=dict(size=12),
        marker=dict(
            size=sizes_raw,        # 실제 데이터(비용)를 그대로 사용
            sizemode="area",       # 면적 기준 스케일링
            sizeref=sizeref,       # 최대 지름을 target_px 근처로
            opacity=bubble_opacity,
            line=dict(width=2)     # 경계 강조 (겹칠 때 구분 쉬움)
        ),
        hovertemplate=(
            f"{bubble_text}=%{{text}}<br>"
            f"{bubble_x}=%{{x}}<br>"
            f"{bubble_y}=%{{y}}<br>"
            f"{bubble_size}=%{{marker.size}}"
            "<extra></extra>"
        )
    ))

    fig_layout_defaults(fig_bubble, "제품별 비용·마진·고객 수 분석", font_family)
    fig_bubble.update_xaxes(title_text=bubble_x, zeroline=True, showgrid=True)
    fig_bubble.update_yaxes(title_text=bubble_y, zeroline=True, showgrid=True)

    st.plotly_chart(fig_bubble, use_container_width=True)

# Footer note
st.caption("© 데이터 분석/수집 20년 파이썬 전문가 — Streamlit + Plotly 대시보드")
