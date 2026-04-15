import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime


def calculate_cci(high, low, close, length=20):
    """計算 CCI 指標。"""
    tp = (high + low + close) / 3
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_val = (tp - sma) / (0.015 * mad)
    return cci_val


def calculate_supertrend(high, low, close, period=6, multiplier=0.686):
    """計算 Supertrend，回傳 direction 與 Supertrend 數值。"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    n = len(close)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1: 空頭, -1: 多頭

    for i in range(period, n):
        if basic_ub.iloc[i] < final_ub[i - 1] or close.iloc[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb.iloc[i] > final_lb[i - 1] or close.iloc[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i - 1]

        if supertrend[i - 1] == final_ub[i - 1] and close.iloc[i] <= final_ub[i]:
            direction[i] = 1
        elif supertrend[i - 1] == final_ub[i - 1] and close.iloc[i] > final_ub[i]:
            direction[i] = -1
        elif supertrend[i - 1] == final_lb[i - 1] and close.iloc[i] >= final_lb[i]:
            direction[i] = -1
        elif supertrend[i - 1] == final_lb[i - 1] and close.iloc[i] < final_lb[i]:
            direction[i] = 1

        supertrend[i] = final_lb[i] if direction[i] == -1 else final_ub[i]

    return direction, supertrend


def check_taiwan_stock(symbol, strict_trend=True):
    """檢驗單檔股票是否符合 CCI 雙向轉折 (多單) 條件。"""
    yf_symbol = symbol if symbol.endswith(".TW") or symbol.endswith(".TWO") else f"{symbol}.TW"
    df = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)

    if len(df) < 60:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()

    if close.empty:
        return None

    direction, _ = calculate_supertrend(high, low, close, period=6, multiplier=0.686)
    cci_val = calculate_cci(high, low, close, length=20)
    cci_ma = cci_val.rolling(14).mean()

    current_idx = -1
    prev_idx = -2

    is_green_trend = direction[current_idx] == -1
    cci_cross_up = (
        cci_val.iloc[prev_idx] <= cci_ma.iloc[prev_idx]
        and cci_val.iloc[current_idx] > cci_ma.iloc[current_idx]
    )

    allow_buy = is_green_trend if strict_trend else True
    trigger_new_buy = cci_cross_up and allow_buy

    if trigger_new_buy:
        return {
            "股票代號": symbol.replace(".TW", "").replace(".TWO", ""),
            "最新收盤價": round(float(close.iloc[current_idx]), 2),
            "CCI 數值": round(float(cci_val.iloc[current_idx]), 2),
            "Supertrend 狀態": "🟢 多頭" if is_green_trend else "🔴 空頭",
            "交易日": df.index[-1].strftime("%Y-%m-%d"),
        }
    return None


st.set_page_config(page_title="台股 CCI 雙向轉折掃描", layout="wide")
st.title("📈 台股買點掃描｜CCI 雙向轉折 (終極潔淨版)")

st.sidebar.header("掃描設定")
strict_trend_ui = st.sidebar.checkbox("開啟大趨勢保護 (Supertrend 綠色才買)", value=True)

default_symbols = "2330, 2317, 2454, 2308, 2881, 2882, 2891, 2412, 1216, 2002"
user_input = st.text_area("請輸入要掃描的股票代號 (以逗號分隔)：", value=default_symbols)

if st.button("開始掃描"):
    symbols = [s.strip() for s in user_input.split(",") if s.strip()]
    st.write(f"準備掃描 {len(symbols)} 檔股票...")

    results = []
    prog = st.progress(0)

    for idx, sym in enumerate(symbols, 1):
        try:
            result = check_taiwan_stock(sym, strict_trend=strict_trend_ui)
            if result:
                results.append(result)
        except Exception as e:
            st.toast(f"讀取 {sym} 時發生錯誤: {e}")

        prog.progress(idx / len(symbols))

    if results:
        df_res = pd.DataFrame(results)
        st.success(f"🎉 掃描完成！找到 {len(df_res)} 檔符合【CCI 黃金交叉】條件的股票")
        st.dataframe(df_res, use_container_width=True)

        csv_bytes = df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下載掃描結果 CSV",
            data=csv_bytes,
            file_name=f"twstock_cci_scan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.warning("目前沒有符合條件的標的（大盤可能偏弱，或 CCI 尚未交叉）。")
