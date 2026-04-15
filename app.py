import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ================= 指標函數 (對齊 TradingView 邏輯) =================

def calculate_cci(high, low, close, length=20):
    """計算 CCI 指標"""
    tp = (high + low + close) / 3
    sma = tp.rolling(length).mean()
    # Pandas 較新版本移除了 .mad()，此處使用 lambda 實作 Mean Absolute Deviation
    mad = tp.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_val = (tp - sma) / (0.015 * mad)
    return cci_val

def calculate_supertrend(high, low, close, period=6, multiplier=0.686):
    """計算 Supertrend，回傳 direction (-1為多頭, 1為空頭) 與 Supertrend 數值"""
    # 1. 計算 ATR (TradingView 預設的 RMA / EWM smoothed ATR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    hl2 = (high + low) / 2
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    n = len(close)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n) # 預設 1 (空頭)，-1 (多頭)

    # 必須以迴圈計算 Supertrend，因為當前值相依於前一個值
    for i in range(period, n):
        # 計算 Final Upper Band
        if basic_ub.iloc[i] < final_ub[i-1] or close.iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # 計算 Final Lower Band
        if basic_lb.iloc[i] > final_lb[i-1] or close.iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]

        # 決定趨勢方向
        if supertrend[i-1] == final_ub[i-1] and close.iloc[i] <= final_ub[i]:
            direction[i] = 1 # 繼續空頭
        elif supertrend[i-1] == final_ub[i-1] and close.iloc[i] > final_ub[i]:
            direction[i] = -1 # 轉多頭
        elif supertrend[i-1] == final_lb[i-1] and close.iloc[i] >= final_lb[i]:
            direction[i] = -1 # 繼續多頭
        elif supertrend[i-1] == final_lb[i-1] and close.iloc[i] < final_lb[i]:
            direction[i] = 1 # 轉空頭

        # 設定 Supertrend 線數值
        supertrend[i] = final_lb[i] if direction[i] == -1 else final_ub[i]

    return direction, supertrend

# ================= 核心條件與掃描邏輯 =================

def check_taiwan_stock(symbol, strict_trend=True):
    """檢驗單檔股票是否符合 CCI 雙向轉折 (多單) 條件"""
    # 確保加上台股後綴，例如輸入 2330 -> 2330.TW
    yf_symbol = symbol if symbol.endswith(".TW") or symbol.endswith(".TWO") else f"{symbol}.TW"
    
    # 獲取日 K 線資料 (取半年資料確保指標運算充足)
    df = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)
    
    if len(df) < 60:
        return None # 資料不足

    # 若 yfinance 回傳 MultiIndex 結構（較新版行為），將其展平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].dropna()
    high = df['High'].dropna()
    low = df['Low'].dropna()
    
    if close.empty:
        return None

    # 1. 運算指標
    direction, st_value = calculate_supertrend(high, low, close, period=6, multiplier=0.686)
    cci_val = calculate_cci(high, low, close, length=20)
    cci_ma = cci_val.rolling(14).mean()
    
    # 取得最新一筆 (i=-1) 與前一筆 (i=-2) 狀態
    current_idx = -1
    prev_idx = -2
    
    is_green_trend = direction[current_idx] == -1
    
    # CCI 交叉條件：前一根 CCI 在 MA 之下，且目前 CCI 穿越至 MA 之上
    cci_cross_up = (cci_val.iloc[prev_idx] <= cci_ma.iloc[prev_idx]) and (cci_val.iloc[current_idx] > cci_ma.iloc[current_idx])
    
    # 2. 綜合判斷 (對應 TradingView 的 cond_cci_buy)
    allow_buy = is_green_trend if strict_trend else True
    trigger_new_buy = cci_cross_up and allow_buy
    
    if trigger_new_buy:
        return {
            "股票代號": symbol.replace(".TW", "").replace(".TWO", ""),
            "最新收盤價": round(float(close.iloc[current_idx]), 2),
            "CCI 數值": round(float(cci_val.iloc[current_idx]), 2),
            "Supertrend 狀態": "🟢 多頭" if is_green_trend else "🔴 空頭",
            "交易日": df.index[-1].strftime("%Y-%m-%d")
        }
    return None

# ================= UI 介面 =================

st.set_page_config(page_title="台股 CCI 雙向轉折掃描", layout="wide")
st.title("📈 台股買點掃描｜CCI 雙向轉折 (終極潔淨版)")

st.sidebar.header("掃描設定")
strict_trend_ui = st.sidebar.checkbox("開啟大趨勢保護 (Supertrend 綠色才買)", value=True)

# 預設台灣50成分股範例
default_symbols = "2330, 2317, 2454, 2308, 2881, 2882, 2891, 2412, 1216, 2002"
user_input = st.text_area("請輸入要掃描的股票代號 (以逗號分隔)：", value=default_symbols)

if st.button("開始掃描"):
    # 清理使用者輸入的代號
    symbols = [s.strip() for s in user_input.split(",") if s.strip()]
    st.write(f"準備掃描 {len(symbols)} 檔股票...")
    
    results = []
    prog = st.progress(0)
    
    for idx, sym in enumerate(symbols, 1):
        try:
            r = check_taiwan_stock(sym, strict_trend=strict_trend_ui)
            if r:
                results.append(r)
        except Exception as e:
            st.toast(f"讀取 {sym} 時發生錯誤: {e}")
            pass
            
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
            mime="text/csv"
        )
    else:
        st.warning("目前沒有符合條件的標的（大盤可能偏弱，或 CCI 尚未交叉）。")
