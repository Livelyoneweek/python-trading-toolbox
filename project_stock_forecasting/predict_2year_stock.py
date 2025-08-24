# predict_2year_stock.py
# 실행: uv run streamlit run predict_2year_stock.py

from __future__ import annotations
import warnings, numpy as np, pandas as pd
import streamlit as st
from pykrx import stock
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import plotly.graph_objs as go

warnings.filterwarnings("ignore", message=".*Conversion of an array with ndim > 0 to a scalar.*")

# ───────────────────────── 유틸 ─────────────────────────
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def load_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start, end, code).sort_index()
    # 컬럼 표준화 (있는 것만 매핑)
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

    # 거래대금 없으면 종가×거래량으로 추정 생성
    if "amount" not in df.columns and {"close","volume"} <= set(df.columns):
        df["amount"] = (df["close"] * df["volume"]).astype(float)

    # 최종 체크
    required = {"open","high","low","close","volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise RuntimeError(f"필수 컬럼 누락: {missing}")

    return df

# 가격 기반 피처(재귀 예측 시 매 스텝 재계산 가능한 것들)
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

# 거래량/거래대금 기반 피처(학습 시엔 전체 시계열, 예측 시엔 스냅샷 고정값)
def make_volume_features_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # amount가 없으면 여기서도 보강
    if "amount" not in df.columns and {"close","volume"} <= set(df.columns):
        df["amount"] = (df["close"] * df["volume"]).astype(float)

    vol = df["volume"].astype(float)
    amt = df["amount"].astype(float)
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    out = pd.DataFrame(index=df.index)

    # 1) RVOL20
    out["rvol20"] = vol / vol.rolling(20).mean()

    # 2) 거래대금 Z-score(20)
    amt_ma = amt.rolling(20).mean()
    amt_sd = amt.rolling(20).std().replace(0, np.nan)
    out["amt_z20"] = (amt - amt_ma) / amt_sd

    # 3) OBV 기울기(5)
    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * vol.fillna(0)).cumsum()
    out["obv_slope5"] = obv.diff(5)

    # 4) MFI14
    tp = (high + low + close) / 3.0
    pmf = pd.Series(np.where(tp > tp.shift(1), tp * vol, 0.0), index=df.index).rolling(14).sum()
    nmf = pd.Series(np.where(tp < tp.shift(1), tp * vol, 0.0), index=df.index).rolling(14).sum().replace(0, np.nan)
    mfr = pmf / nmf
    out["mfi14"] = (100 - 100 / (1 + mfr)) / 100.0  # 0~1

    return out.replace([np.inf, -np.inf], np.nan)

def last_volume_snapshot(df: pd.DataFrame) -> dict[str, float]:
    vol_feats = make_volume_features_series(df).dropna(how="any")
    if vol_feats.empty:
        # 링 윈도우 초기 구간 등으로 비면 기본값 사용
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

def bdays_after(base: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(base + pd.Timedelta(days=1), periods=periods, freq="B")

def smape(y_true, y_pred, eps=1e-12):
    a = np.abs(y_pred - y_true)
    b = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return float(np.mean(2 * a / b))

# ───────────────────────── UI ─────────────────────────
st.set_page_config(page_title="005930 15D 예측 (RF + Volume)", layout="wide")
st.title("삼성전자(005930) 15일 예측 – RF (가격+거래량 피처)")

with st.sidebar:
    st.markdown("### 설정")
    horizon = st.number_input("예측 영업일 수", 5, 30, 15, 1)
    years   = st.slider("학습 기간(년)", 1, 10, 2, 1)
    n_est   = st.slider("트리 수", 100, 1000, 300, 50)
    max_depth = st.selectbox("최대 깊이", [None,4,6,8,10], index=2)
    n_sims  = st.slider("시뮬 경로", 100, 2000, 300, 100)
    seed    = st.number_input("랜덤시드", 0, 99999, 42, 1)
    fast    = st.checkbox("초고속 모드(품질↓)", value=False)
    predict_clicked = st.button("예측하기", type="primary")

if fast:
    n_est = min(n_est, 200)
    n_sims = min(n_sims, 150)

# 첫 실행은 자동 예측 1회, 그 후엔 버튼으로만
if "auto_predict_once" not in st.session_state:
    st.session_state.auto_predict_once = True

# ───────────────────── 데이터/학습 ─────────────────────
try:
    with st.status("데이터 로딩...", expanded=False) as s:
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=365*years + 60)).strftime("%Y%m%d")  # 여유 버퍼
        code = "005930"
        df = load_ohlcv(code, start, today)
        if df.empty:
            raise RuntimeError("pykrx에서 데이터가 비었습니다.")
        st.write("데이터:", df.shape)
        s.update(label="데이터 OK", state="complete")

    with st.status("피처/레이블 생성...", expanded=False) as s:
        X_all = make_features_train(df)
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

        rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, random_state=int(seed), n_jobs=-1)
        rf.fit(X_tr, y_tr)

        yhat_va = rf.predict(X_va)
        resid_std = float(np.nanstd(y_va.values - yhat_va))
        if not np.isfinite(resid_std) or resid_std <= 0:
            resid_std = float(np.nanstd(y_tr.values - rf.predict(X_tr))) or 0.02

        mae = float(mean_absolute_error(y_va, yhat_va))
        smp = smape(y_va.values, yhat_va)
        dir_acc = float((np.sign(y_va.values) == np.sign(yhat_va)).mean())

        st.write(f"검증 MAE: {mae:.4f}  |  SMAPE: {smp:.2%}  |  방향정확도: {dir_acc:.1%}")
        s.update(label="학습 완료", state="complete")

    # ───────── 예측 트리거 ─────────
    do_predict = predict_clicked or st.session_state.auto_predict_once
    if st.session_state.auto_predict_once:
        st.session_state.auto_predict_once = False

    if not do_predict:
        st.info("사이드바에서 **예측하기** 버튼을 눌러 예측을 시작하세요.", icon="➡️")
        st.stop()

    # ───────── 예측(재귀 + 몬테카를로) ─────────
    np.random.seed(int(seed))
    last_idx = X_all.index[-1]
    future_idx = bdays_after(last_idx, int(horizon))
    tail_len = 120  # 가격 피처 재계산 최소 구간
    vol_fix = last_volume_snapshot(df)  # ← A 방식: 스냅샷 고정

    def make_features_forecast_row(close_series: pd.Series) -> pd.DataFrame:
        Xp = make_price_features(close_series).iloc[[-1]]
        # 고정 볼륨 피처 주입(학습 컬럼과 동일 이름)
        for k, v in vol_fix.items():
            Xp[k] = v
        # 학습 컬럼 정합성 확보: 학습에 있던 컬럼만 사용
        return Xp[X_all.columns]

    def forecast(close_series: pd.Series, steps: int, sims: int, noise_std: float):
        # 결정론 경로
        c = close_series.copy()
        det_vals = []
        for _ in range(steps):
            X_t = make_features_forecast_row(c.tail(tail_len))
            mu = float(rf.predict(X_t)[0])
            nxt = float(c.iloc[-1] * (1 + mu))
            c = pd.concat([c, pd.Series([nxt], index=[c.index[-1] + pd.Timedelta(days=1)])])
            det_vals.append(nxt)
        det = pd.Series(det_vals, index=future_idx, name="det")

        # 시뮬레이션
        paths = np.empty((steps, sims), dtype=float)
        prog = st.progress(0.0)
        for j in range(sims):
            cs = close_series.copy()
            for i in range(steps):
                X_t = make_features_forecast_row(cs.tail(tail_len))
                mu = float(rf.predict(X_t)[0])
                r_next = mu + np.random.normal(0.0, noise_std)
                nxt = float(cs.iloc[-1] * (1 + r_next))
                cs = pd.concat([cs, pd.Series([nxt], index=[cs.index[-1] + pd.Timedelta(days=1)])])
                paths[i, j] = nxt
            if (j+1) % max(1, sims//100) == 0:
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

    x_band = np.concatenate([pred_low.index.values, pred_high.index.values[::-1]])
    y_band = np.concatenate([pred_low.values,       pred_high.values[::-1]])
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill='toself', fillcolor='rgba(100,100,200,0.15)',
                             line=dict(color='rgba(100,100,200,0)'), name='10~90% 밴드'))
    fig.update_layout(title="삼성전자 – 다음 15영업일 예측 (RF 재귀 + MC, 거래량 스냅샷 고정)",
                      xaxis_title="날짜", yaxis_title="가격(원)",
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # ───────── 예측 상세(리스트/표/다운로드) ─────────
    st.markdown("### 예측 상세 (평균 경로)")
    pred_lines = [f"- {d.strftime('%Y/%m/%d')} 예측종가: {v:,.0f}원" for d, v in pred_mean.items()]
    st.markdown("\n".join(pred_lines))

    pred_df = pd.DataFrame({
        "예측일": pred_mean.index.strftime("%Y/%m/%d"),
        "예측종가(평균)": pred_mean.round(0).astype(int),
        "예측종가(결정론)": det_series.reindex(pred_mean.index).round(0).astype(int),
        "하단밴드(10%)": pred_low.round(0).astype(int),
        "상단밴드(90%)": pred_high.round(0).astype(int),
    })
    st.dataframe(pred_df, use_container_width=True)
    st.download_button(
        "예측표 CSV 다운로드",
        pred_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="forecast_005930_15d.csv",
        mime="text/csv",
    )

    # ───────── 요약/검증 ─────────
    st.subheader("요약")
    last_p = float(hist.iloc[-1]); end_mean = float(pred_mean.iloc[-1])
    chg = (end_mean/last_p - 1)*100
    st.write(f"- 마지막 종가: **{last_p:,.0f}원**")
    st.write(f"- 예측 종료일 평균가격(시뮬): **{end_mean:,.0f}원** ({chg:+.2f}%)")

except Exception as e:
    st.exception(e)