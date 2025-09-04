# predict_2year_stock.py
# 실행: uv run streamlit run predict_2year_stock.py

from __future__ import annotations
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from numpy.random import default_rng
from pykrx import stock
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go

warnings.filterwarnings("ignore", message=".*Conversion of an array with ndim > 0 to a scalar.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# ───────────────────────── 유틸 ─────────────────────────
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def load_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start, end, code).sort_index()

    # 컬럼 표준화
    rename_map = {}
    for src, dst in [("시가","open"),("고가","high"),("저가","low"),
                     ("종가","close"),("거래량","volume"),("거래대금","amount")]:
        if src in df.columns:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    # 타입 정리
    for c in ["open","high","low","close","volume","amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 거래대금 없으면 종가×거래량으로 보강
    if "amount" not in df.columns and {"close","volume"} <= set(df.columns):
        df["amount"] = (df["close"] * df["volume"]).astype(float)

    required = {"open","high","low","close","volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise RuntimeError(f"필수 컬럼 누락: {missing}")
    return df

def make_price_features(close: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=close.index)
    X["r1"] = close.pct_change(1)
    X["r3"] = close.pct_change(3)
    X["r5"] = close.pct_change(5)
    X["r10"] = close.pct_change(10)
    for n in (5,10,20,60):
        ma = close.rolling(n).mean()
        std = close.rolling(n).std()
        X[f"gap_ma{n}"] = close/ma - 1
        X[f"bb_pos{n}"] = (close-ma)/std.replace(0,np.nan)
    X["rsi14"] = (rsi(close)/100).clip(0,1)
    return X.replace([np.inf,-np.inf], np.nan)

def make_volume_features_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "amount" not in df.columns and {"close","volume"} <= set(df.columns):
        df["amount"] = (df["close"] * df["volume"]).astype(float)

    vol = df["volume"].astype(float)
    amt = df["amount"].astype(float)
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["rvol20"] = vol / vol.rolling(20).mean()

    amt_ma = amt.rolling(20).mean()
    amt_sd = amt.rolling(20).std().replace(0, np.nan)
    out["amt_z20"] = (amt - amt_ma) / amt_sd

    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * vol.fillna(0)).cumsum()
    out["obv_slope5"] = obv.diff(5)

    tp = (high + low + close) / 3.0
    pmf = pd.Series(np.where(tp > tp.shift(1), tp * vol, 0.0), index=df.index).rolling(14).sum()
    nmf = pd.Series(np.where(tp < tp.shift(1), tp * vol, 0.0), index=df.index).rolling(14).sum().replace(0, np.nan)
    mfr = pmf / nmf
    out["mfi14"] = (100 - 100 / (1 + mfr)) / 100.0

    return out.replace([np.inf, -np.inf], np.nan)

def last_volume_snapshot(df: pd.DataFrame) -> dict[str, float]:
    vol_feats = make_volume_features_series(df).dropna(how="any")
    if vol_feats.empty:
        return {"rvol20": 1.0, "amt_z20": 0.0, "obv_slope5": 0.0, "mfi14": 0.5}
    s = vol_feats.iloc[-1]
    return {
        "rvol20": float(s.get("rvol20", 1.0)),
        "amt_z20": float(s.get("amt_z20", 0.0)),
        "obv_slope5": float(s.get("obv_slope5", 0.0)),
        "mfi14": float(s.get("mfi14", 0.5)),
    }

def make_features_train(df: pd.DataFrame) -> pd.DataFrame:
    Xp = make_price_features(df["close"])
    Xv = make_volume_features_series(df)
    X = Xp.join(Xv).dropna()
    return X

def make_target_next_return(close: pd.Series) -> pd.Series:
    return close.shift(-1)/close - 1

def smape(y_true, y_pred, eps=1e-12):
    a = np.abs(y_pred - y_true)
    b = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2 * a / b))

def next_trading_days(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """
    미래 거래일 n개 생성.
    holidays 패키지가 있으면 공휴일까지 제외, 없으면 주말만 제외.
    """
    try:
        import holidays as pyhol
        kr_holidays = pyhol.KR(years=[last_date.year, last_date.year + 1])
        days = []
        d = last_date
        while len(days) < n:
            d = d + pd.Timedelta(days=1)
            if d.weekday() < 5 and d.date() not in kr_holidays:
                days.append(pd.Timestamp(d.normalize()))
        return pd.DatetimeIndex(days)
    except Exception:
        return pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n)

# ───────────────────────── Streamlit ─────────────────────────
st.set_page_config(page_title="주가 15영업일 예측 (RF + Volume)", layout="wide")
st.title("주가 15영업일 예측 – RF (가격+거래량 피처)")

with st.sidebar:
    st.markdown("### 설정")
    code = st.text_input("종목코드 (예: 005930)", value="005930", help="예시로 삼성전자(005930)를 기본값으로 넣어둠")
    horizon = st.number_input("예측 영업일 수", 5, 30, 15, 1)
    years   = st.slider("학습 기간(년)", 1, 10, 2, 1)
    n_est   = st.slider("트리 수", 100, 1000, 300, 50)
    max_depth = st.selectbox("최대 깊이", [None,4,6,8,10], index=2)  # 기본 6
    n_sims  = st.slider("시뮬 경로 수", 50, 3000, 300, 50)
    seed    = st.number_input("랜덤시드", 0, 99999, 42, 1)
    fast    = st.checkbox("초고속 모드(품질↓)", value=False)
    use_decay = st.checkbox("거래량 스냅샷 감쇠(현실감↑)", value=False)

    # (3) 결정론 μ 기반 벡터화 선택
    use_mu_vectorized = st.checkbox("결정론 μ 기반 벡터화 시뮬(가속)", value=False,
                                    help="체크 시 결정론 경로에서 구한 μ 시퀀스를 고정하고 노이즈만 섞어 벡터화 계산(매우 빠름)")

    # (4-A) 샘플 경로 표시 옵션 (표시만)
    show_sample_paths = st.checkbox("샘플 경로 추가 표시", value=True,
                                    help="시뮬레이션 결과에서 무작위 샘플 경로 몇 개를 그래프에 같이 표시합니다(계산에 영향 없음).")
    sample_count = st.slider("샘플 경로 개수", 1, 10, 5, 1, disabled=not show_sample_paths)

    # (4-B) 실현 변동성 기반 노이즈 사용(계산에 영향)
    use_realized_vol = st.checkbox("실현 변동성 기반 노이즈 사용", value=True,
                                   help="잔차표준편차 대신(or 함께) 최근 수익률 변동성을 반영하여 일별 변동 폭을 현실적으로 만듭니다.")
    realized_scale = st.slider("실현 변동성 가중치", 0.1, 2.0, 0.7, 0.1, disabled=not use_realized_vol)

    predict_clicked = st.button("예측하기", type="primary")

if fast:
    n_est = min(n_est, 200)
    n_sims = min(n_sims, 200)

# ───────────────────────── 캐시 ─────────────────────────
@st.cache_data(ttl=60*10, show_spinner=False)
def load_ohlcv_cached(code: str, start: str, end: str) -> pd.DataFrame:
    return load_ohlcv(code, start, end)

@st.cache_data(ttl=60*10, show_spinner=False)
def build_features_cached(df: pd.DataFrame) -> pd.DataFrame:
    return make_features_train(df)

# ───────────────────── 데이터/학습 ─────────────────────
try:
    # (2) 처음 시작 시 자동 예측 금지 → 버튼 누르기 전엔 바로 stop
    if not predict_clicked:
        st.info("사이드바에서 종목코드와 옵션을 설정한 뒤 **예측하기** 버튼을 눌러주세요.", icon="➡️")
        st.stop()

    with st.status("데이터 로딩...", expanded=False) as s:
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=365*years + 60)).strftime("%Y%m%d")
        df = load_ohlcv_cached(code, start, today)
        if df.empty:
            raise RuntimeError("pykrx에서 데이터가 비었습니다.")
        st.write("데이터:", df.shape)
        s.update(label="데이터 OK", state="complete")

    with st.status("피처/레이블 생성...", expanded=False) as s:
        X_all = build_features_cached(df)
        y_all = make_target_next_return(df["close"]).loc[X_all.index]
        XY = pd.concat([X_all, y_all.rename("y")], axis=1).dropna()
        if len(XY) < 200:
            raise RuntimeError("학습 데이터가 부족합니다. 기간을 늘려보세요.")
        st.write("학습셋:", XY.shape)
        s.update(label="피처 OK", state="complete")

    with st.status("모델 학습...", expanded=False) as s:
        cut = int(len(XY) * 0.8)
        tr, va = XY.iloc[:cut], XY.iloc[cut:]
        X_tr, y_tr = tr.drop(columns="y"), tr["y"]
        X_va, y_va = va.drop(columns="y"), va["y"]

        rf = RandomForestRegressor(
            n_estimators=int(n_est),
            max_depth=None if max_depth is None else int(max_depth),
            random_state=int(seed),
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)

        yhat_va = rf.predict(X_va)
        resid_std = float(np.nanstd(y_va.values - yhat_va))
        if not np.isfinite(resid_std) or resid_std <= 0:
            alt = np.nanstd(y_tr.values - rf.predict(X_tr))
            resid_std = float(alt) if np.isfinite(alt) and alt > 0 else 0.02

        # (4-B) 실현 변동성 기반 노이즈: 계산에 영향 있음
        if use_realized_vol:
            real_sigma = float(df["close"].pct_change().dropna().tail(60).std())
            resid_std = max(resid_std, real_sigma * float(realized_scale))

        mae = float(mean_absolute_error(y_va, yhat_va))
        smp = smape(y_va.values, yhat_va)
        dir_acc = float((np.sign(y_va.values) == np.sign(yhat_va)).mean())
        st.write(f"검증 MAE: {mae:.4f}  |  SMAPE: {smp:.2%}  |  방향정확도: {dir_acc:.1%}")
        s.update(label="학습 완료", state="complete")


        # ───────── 검증 결과 시각화 3종 세트 ─────────
        st.subheader("검증 결과 시각화")

        # 0) 준비
        va_index = y_va.index
        y_true = y_va.values          # 실제 일수익률
        y_pred = yhat_va              # 예측 일수익률 (numpy array or Series)

        # 1) 실제 vs 예측 (산점도)
        st.markdown("#### 1) 실제 vs 예측 (산점도)")
        x_min = float(np.nanmin(y_true))
        x_max = float(np.nanmax(y_true))
        pad   = (x_max - x_min) * 0.05 if x_max > x_min else 0.01
        line_x = np.linspace(x_min - pad, x_max + pad, 100)

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers", name="검증 샘플",
            marker=dict(size=6, opacity=0.6)))
        fig_scatter.add_trace(go.Scatter(
            x=line_x, y=line_x, mode="lines", name="y = x",
            line=dict(color="red", dash="dash")))
        fig_scatter.update_layout(
            xaxis_title="실제 일수익률",
            yaxis_title="예측 일수익률",
            hovermode="closest"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 2) 검증 구간 시계열(일수익률 비교)
        st.markdown("#### 2) 검증 구간 시계열 (일수익률)")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=va_index, y=y_true, name="실제 일수익률", mode="lines"))
        fig_ts.add_trace(go.Scatter(x=va_index, y=y_pred, name="예측 일수익률", mode="lines"))
        fig_ts.update_layout(
            xaxis_title="날짜",
            yaxis_title="일수익률",
            hovermode="x unified"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # (옵션) 누적수익률 비교도 보고 싶다면 주석 해제
        # cum_true = pd.Series(1 + y_va.values, index=va_index).cumprod()
        # cum_pred = pd.Series(1 + yhat_va,     index=va_index).cumprod()
        # fig_cum = go.Figure()
        # fig_cum.add_trace(go.Scatter(x=va_index, y=cum_true, name="실제 누적수익률", mode="lines"))
        # fig_cum.add_trace(go.Scatter(x=va_index, y=cum_pred, name="예측 누적수익률", mode="lines"))
        # fig_cum.update_layout(xaxis_title="날짜", yaxis_title="누적수익률지수(=1 기준)", hovermode="x unified")
        # st.plotly_chart(fig_cum, use_container_width=True)

        # 3) 오차 분포 (히스토그램)
        st.markdown("#### 3) 오차 분포 (y_true - y_pred)")
        errors = y_true - y_pred
        fig_hist = go.Figure(data=[go.Histogram(x=errors, nbinsx=40, name="오차")])
        fig_hist.update_layout(
            xaxis_title="오차 (실제 - 예측)",
            yaxis_title="빈도",
            bargap=0.02
        )
        st.plotly_chart(fig_hist, use_container_width=True)



    # ───────── 예측 준비 ─────────
    rng = default_rng(int(seed))
    last_idx = X_all.index[-1]
    future_idx = next_trading_days(last_idx, int(horizon))
    if len(future_idx) < int(horizon):
        st.warning("미래 거래일 생성이 부족하여 주말만 제외한 기준으로 보정했습니다.")
        future_idx = pd.bdate_range(last_idx + pd.Timedelta(days=1), periods=int(horizon))

    tail_len = 60 if fast else 120
    base_vol_fix = last_volume_snapshot(df)

    neutrals = {"rvol20":1.0, "amt_z20":0.0, "obv_slope5":0.0, "mfi14":0.5}
    decay_lambda = 0.15

    feature_median = X_tr.median(numeric_only=True)
    all_columns = X_all.columns

    def fill_and_align(Xp: pd.DataFrame) -> pd.DataFrame:
        Xp = Xp.reindex(columns=all_columns)
        Xp = Xp.fillna(feature_median)
        Xp = Xp.fillna(0.0)
        return Xp

    def make_features_forecast_row(close_series: pd.Series, vol_state: dict[str, float]) -> pd.DataFrame:
        Xp = make_price_features(close_series).iloc[[-1]]
        for k, v in vol_state.items():
            Xp[k] = v
        return fill_and_align(Xp)

    def decay_to_neutral(v: float, neutral: float, lam: float) -> float:
        return float(v + (neutral - v) * lam)

    # ───────── 예측(결정론 + 시뮬) ─────────
    def forecast(close_series: pd.Series, steps: int, sims: int, noise_std: float):
        # 결정론 경로
        c = close_series.copy()
        det_vals = []
        mu_seq = []  # 결정론 경로에서 얻은 μ 시퀀스
        vol_state_det = base_vol_fix.copy()
        for i in range(steps):
            X_t = make_features_forecast_row(c.tail(tail_len), vol_state_det)
            mu = float(rf.predict(X_t)[0])
            mu_seq.append(mu)
            nxt = float(c.iloc[-1] * (1 + mu))
            c.loc[future_idx[i]] = nxt
            det_vals.append(nxt)
            if use_decay:
                for k, neutral in neutrals.items():
                    vol_state_det[k] = decay_to_neutral(vol_state_det.get(k, neutral), neutral, decay_lambda)
        det = pd.Series(det_vals, index=future_idx, name="det")

        # 시뮬레이션
        if use_mu_vectorized:
            mu_arr = np.array(mu_seq, dtype=float)[:, None]             # (steps, 1)
            noise = rng.normal(0.0, noise_std, size=(steps, sims))      # (steps, sims)
            r_seq = mu_arr + noise                                      # (steps, sims)

            paths = np.empty((steps, sims), dtype=float)
            prices = np.full(sims, close_series.iloc[-1], dtype=float)
            for i in range(steps):
                prices *= (1.0 + r_seq[i])
                paths[i] = prices
        else:
            paths = np.empty((steps, sims), dtype=float)
            prog = st.progress(0.0)
            update_every = max(1, sims // 20)
            for j in range(sims):
                cs = close_series.copy()
                vol_state = base_vol_fix.copy()
                for i in range(steps):
                    X_t = make_features_forecast_row(cs.tail(tail_len), vol_state)
                    mu = float(rf.predict(X_t)[0])
                    r_next = mu + float(rng.normal(0.0, noise_std))
                    nxt = float(cs.iloc[-1] * (1 + r_next))
                    cs.loc[future_idx[i]] = nxt
                    paths[i, j] = nxt
                    if use_decay:
                        for k, neutral in neutrals.items():
                            vol_state[k] = decay_to_neutral(vol_state.get(k, neutral), neutral, decay_lambda)
                if (j+1) % update_every == 0:
                    prog.progress((j+1)/sims)
            prog.empty()

        return det, paths

    with st.status("예측 중...", expanded=False) as s:
        det_series, sim_paths = forecast(df["close"], int(horizon), int(n_sims), resid_std)
        s.update(label="예측 완료", state="complete")

    sim_df = pd.DataFrame(sim_paths, index=future_idx).T  # (sims x steps)
    mean_path = sim_df.mean(axis=0)
    p10 = sim_df.quantile(0.10, axis=0)
    p90 = sim_df.quantile(0.90, axis=0)

    hist = df["close"].iloc[-120:]
    pred_mean = pd.Series(mean_path.values, index=future_idx, name="pred_mean")
    pred_low  = pd.Series(p10.values,       index=future_idx, name="pred_p10")
    pred_high = pd.Series(p90.values,       index=future_idx, name="pred_p90")

    # ───────── Plot ─────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="과거 종가", mode="lines"))
    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, name="예측 평균(시뮬)", mode="lines"))
    fig.add_trace(go.Scatter(x=det_series.index, y=det_series.values, name="예측 경로(결정론)", mode="lines", line=dict(dash="dot")))

    # (4-A) 샘플 경로 표시: 계산엔 영향 없음, 그래프에만 추가
    if show_sample_paths and len(sim_df) >= 1:
        rng_vis = default_rng(int(seed) + 123)
        sample_n = min(sample_count, sim_df.shape[0])
        rows = rng_vis.choice(sim_df.index.values, size=sample_n, replace=False)
        for r in rows:
            fig.add_trace(go.Scatter(
                x=future_idx, y=sim_df.loc[r].values,
                name=f"샘플경로#{int(r)}", mode="lines",
                line=dict(width=1), opacity=0.45
            ))

    x_band = np.concatenate([pred_low.index.values, pred_high.index.values[::-1]])
    y_band = np.concatenate([pred_low.values,       pred_high.values[::-1]])
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill='toself', fillcolor='rgba(100,100,200,0.15)',
                             line=dict(color='rgba(100,100,200,0)'), name='10~90% 밴드'))
    fig.update_layout(
        title=f"{code} – 다음 {int(horizon)}영업일 예측 "
              f"(RF 재귀 + MC, 거래량 스냅샷{' 감쇠' if use_decay else ' 고정'}"
              f"{' | μ-벡터화' if use_mu_vectorized else ''}"
              f"{' | 실현변동성노이즈' if use_realized_vol else ''})",
        xaxis_title="날짜", yaxis_title="가격(원)",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ───────── 예측 상세(리스트/표/다운로드) ─────────
    st.markdown("### 예측 상세 (평균 경로)")
    pred_lines = [f"- {d.strftime('%Y/%m/%d')} 예측종가: {v:,.0f}원" for d, v in pred_mean.items()]
    st.markdown("\n".join(pred_lines))

    pred_df = pd.DataFrame({
        "예측일": pred_mean.index.strftime("%Y/%m/%d"),
        "예측종가(평균)": pred_mean.round(0).astype(int),
        "예측종가(결정론)": pd.Series(det_series).reindex(pred_mean.index).round(0).astype(int),
        "하단밴드(10%)": pred_low.round(0).astype(int),
        "상단밴드(90%)": pred_high.round(0).astype(int),
    })
    st.dataframe(pred_df, use_container_width=True)
    st.download_button(
        "예측표 CSV 다운로드",
        pred_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"forecast_{code}_{int(horizon)}d.csv",
        mime="text/csv",
    )

    # ───────── 요약 ─────────
    st.subheader("요약")
    last_p = float(hist.iloc[-1]); end_mean = float(pred_mean.iloc[-1])
    chg = (end_mean/last_p - 1)*100
    st.write(f"- 마지막 종가: **{last_p:,.0f}원**")
    st.write(f"- 예측 종료일 평균가격(시뮬): **{end_mean:,.0f}원** ({chg:+.2f}%)")

except Exception as e:
    st.exception(e)
