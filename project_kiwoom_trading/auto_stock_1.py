import pandas as pd
from datetime import datetime
from pykrx import stock

###############################################################################
# 1. 데이터 다운로드 ─ 2023-01-02 ~ 2023-12-29 삼성전자(005930)
###############################################################################
start, end = "2023-01-02", "2023-12-29"
raw = stock.get_market_ohlcv_by_date(start, end, "005930")
# 컬럼명 한글 → 영문으로 변경
df = raw.rename(columns={
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume"
})
df.index = pd.to_datetime(df.index)

###############################################################################
# 2. 변동성 돌파 로직 정의 (변경 없음)
###############################################################################
def volatility_breakout(data: pd.DataFrame, k: float = 0.5) -> pd.DataFrame:
    df = data.copy()
    df["range"]  = (df["high"].shift(1) - df["low"].shift(1)) * k
    df["target"] = df["open"] + df["range"]
    df["buy"]    = df["high"] >= df["target"]
    df["ret"]    = 0.0
    buy_idx      = df["buy"]
    df.loc[buy_idx, "ret"] = df.loc[buy_idx, "close"] / df.loc[buy_idx, "open"] - 1
    df["cum_ret"] = (1 + df["ret"]).cumprod()
    return df

###############################################################################
# 3. 백테스트 실행 & 핵심 지표 출력
###############################################################################
k  = 0.5
bt = volatility_breakout(df, k)

days     = (bt.index[-1] - bt.index[0]).days
cagr     = bt["cum_ret"].iloc[-1] ** (365 / days) - 1
dd       = bt["cum_ret"] / bt["cum_ret"].cummax() - 1
mdd      = dd.min()
win_rate = (bt["ret"] > 0).sum() / (bt["ret"] != 0).sum()

print(f"[Volatility Breakout] k={k:.2f}")
print(f"누적수익률   : {bt['cum_ret'].iloc[-1]:.2f}배")
print(f"CAGR         : {cagr:.2%}")
print(f"최대낙폭(MDD) : {mdd:.2%}")
print(f"승률         : {win_rate:.2%} (체결 { (bt['ret']!=0).sum() }일)")

###############################################################################
# 4. k 값 스윕(0.1 ~ 1.0) ─ 파라미터 최적화 예시
###############################################################################
def sweep_k(k_list):
    result = []
    for k in k_list:
        final = volatility_breakout(df, k)["cum_ret"].iloc[-1]
        result.append((k, final))
    return pd.DataFrame(result, columns=["k", "final_multiplier"]).set_index("k")

k_grid = [round(x / 10, 1) for x in range(1, 11)]
print(sweep_k(k_grid))