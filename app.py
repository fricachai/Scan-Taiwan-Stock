import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import json
import re
from datetime import datetime
from urllib.error import URLError
from urllib.request import urlopen


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


@st.cache_data(show_spinner=False)
def fetch_twse_stock_name(symbol):
    """從 TWSE 月資料 API 的 title 欄位拆出股票名稱。"""
    code = symbol.replace(".TW", "").replace(".TWO", "")
    month_starts = [
        pd.Timestamp.today().replace(day=1),
        pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=1),
    ]

    for month_start in month_starts:
        api_date = month_start.strftime("%Y%m01")
        url = (
            "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
            f"?response=json&date={api_date}&stockNo={code}"
        )

        try:
            with urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError):
            continue

        if payload.get("stat") != "OK":
            continue

        title = " ".join(str(payload.get("title", "")).split())
        match = re.search(rf"{re.escape(code)}\s+(.+)", title)
        if not match:
            continue

        stock_name = next((part for part in match.group(1).split(" ") if part), "")
        if stock_name:
            return stock_name

    return ""


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
        stock_code = symbol.replace(".TW", "").replace(".TWO", "")
        return {
            "股票代號": stock_code,
            "股票名稱": fetch_twse_stock_name(stock_code),
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

default_symbols = "1101,1102,1103,1104,1108,1109,1110,1201,1203,1210,1213,1215,1216,1217,1218,1219,1220,1225,1227,1229,1231,1232,1233,1234,1235,1236,1256,1301,1303,1304,1305,1307,1308,1309,1310,1312,1313,1314,1315,1316,1319,1321,1323,1324,1325,1326,1337,1338,1339,1340,1341,1342,1402,1409,1410,1413,1414,1416,1417,1418,1419,1423,1432,1434,1435,1436,1437,1438,1439,1440,1441,1442,1443,1444,1445,1446,1447,1449,1451,1452,1453,1454,1455,1456,1457,1459,1460,1463,1464,1465,1466,1467,1468,1470,1471,1472,1473,1474,1475,1476,1477,1503,1504,1506,1512,1513,1514,1515,1516,1517,1519,1521,1522,1524,1525,1526,1527,1528,1529,1530,1531,1532,1533,1535,1536,1537,1538,1539,1540,1541,1558,1560,1563,1568,1582,1583,1587,1589,1590,1597,1598,1603,1604,1605,1608,1609,1611,1612,1614,1615,1616,1617,1618,1623,1626,1702,1707,1708,1709,1710,1711,1712,1713,1714,1717,1718,1720,1721,1722,1723,1725,1726,1727,1730,1731,1732,1733,1734,1735,1736,1737,1752,1760,1762,1773,1776,1783,1786,1789,1795,1802,1805,1806,1808,1809,1810,1817,1903,1904,1905,1906,1907,1909,2002,2006,2007,2008,2009,2010,2012,2013,2014,2015,2017,2020,2022,2023,2024,2025,2027,2028,2029,2030,2031,2032,2033,2034,2038,2049,2059,2062,2069,2072,2101,2102,2103,2104,2105,2106,2107,2108,2109,2114,2115,2201,2204,2206,2207,2208,2211,2227,2228,2231,2233,2236,2239,2241,2243,2247,2248,2250,2254,2258,2301,2302,2303,2305,2308,2312,2313,2314,2316,2317,2321,2323,2324,2327,2328,2329,2330,2331,2332,2337,2338,2340,2342,2344,2345,2347,2348,2349,2351,2352,2353,2354,2355,2356,2357,2359,2360,2362,2363,2364,2365,2367,2368,2369,2371,2373,2374,2375,2376,2377,2379,2380,2382,2383,2385,2387,2388,2390,2392,2393,2395,2397,2399,2401,2402,2404,2405,2406,2408,2409,2412,2413,2414,2415,2417,2419,2420,2421,2423,2424,2425,2426,2427,2428,2429,2430,2431,2432,2433,2434,2436,2438,2439,2440,2441,2442,2444,2449,2450,2451,2453,2454,2455,2457,2458,2459,2460,2461,2462,2464,2465,2466,2467,2468,2471,2472,2474,2476,2477,2478,2480,2481,2482,2483,2484,2485,2486,2488,2489,2491,2492,2493,2495,2496,2497,2498,2501,2504,2505,2506,2509,2511,2514,2515,2516,2520,2524,2527,2528,2530,2534,2535,2536,2537,2538,2539,2540,2542,2543,2545,2546,2547,2548,2597,2601,2603,2605,2606,2607,2608,2609,2610,2611,2612,2613,2614,2615,2616,2617,2618,2630,2633,2634,2636,2637,2642,2645,2646,2701,2702,2704,2705,2706,2707,2712,2722,2723,2727,2731,2739,2748,2753,2762,2801,2812,2816,2820,2832,2834,2836,2838,2845,2849,2850,2851,2852,2855,2867,2880,2881,2882,2883,2884,2885,2886,2887,2889,2890,2891,2892,2897,2901,2903,2904,2905,2906,2908,2910,2911,2912,2913,2915,2923,2929,2939,2945,3002,3003,3004,3005,3006,3008,3010,3011,3013,3014,3015,3016,3017,3018,3019,3021,3022,3023,3024,3025,3026,3027,3028,3029,3030,3031,3032,3033,3034,3035,3036,3037,3038,3040,3041,3042,3043,3044,3045,3046,3047,3048,3049,3050,3051,3052,3054,3055,3056,3057,3058,3059,3060,3062,3090,3092,3094,3130,3135,3138,3149,3150,3164,3167,3168,3189,3209,3229,3231,3257,3266,3296,3305,3308,3311,3312,3321,3338,3346,3356,3376,3380,3406,3413,3416,3419,3432,3437,3443,3447,3450,3481,3494,3501,3504,3515,3518,3528,3530,3532,3533,3535,3543,3545,3550,3557,3563,3576,3583,3588,3591,3592,3593,3596,3605,3607,3617,3622,3645,3652,3653,3661,3665,3669,3673,3679,3686,3694,3701,3702,3703,3704,3705,3706,3708,3711,3712,3714,3715,3716,3717,4104,4106,4108,4119,4133,4137,4142,4148,4155,4164,4169,4190,4306,4414,4426,4438,4439,4440,4441,4526,4532,4536,4540,4545,4551,4552,4555,4557,4560,4562,4564,4566,4569,4571,4572,4576,4581,4583,4585,4588,4590,4720,4722,4736,4737,4739,4746,4755,4763,4764,4766,4770,4771,4807,4904,4906,4912,4915,4916,4919,4927,4930,4934,4935,4938,4942,4943,4949,4952,4956,4958,4960,4961,4967,4968,4976,4977,4989,4994,4999,5007,5203,5215,5222,5225,5234,5243,5244,5258,5269,5283,5284,5285,5288,5292,5306,5388,5434,5469,5471,5484,5515,5519,5521,5522,5525,5531,5533,5534,5538,5546,5607,5608,5706,5871,5876,5880,5906,5907,6005,6024,6108,6112,6115,6116,6117,6120,6128,6133,6136,6139,6141,6142,6152,6153,6155,6164,6165,6166,6168,6176,6177,6183,6184,6189,6191,6192,6196,6197,6201,6202,6205,6206,6209,6213,6214,6215,6216,6224,6225,6226,6230,6235,6239,6243,6257,6269,6271,6272,6277,6278,6281,6282,6283,6285,6405,6409,6412,6414,6415,6416,6426,6431,6438,6442,6443,6446,6449,6451,6456,6464,6472,6477,6491,6504,6505,6515,6525,6526,6531,6533,6534,6541,6550,6552,6558,6573,6579,6581,6582,6585,6589,6591,6592,6598,6605,6606,6614,6625,6641,6645,6655,6657,6658,6666,6668,6669,6670,6671,6672,6674,6689,6691,6695,6698,6706,6715,6719,6722,6742,6743,6753,6754,6756,6757,6768,6770,6771,6776,6781,6782,6789,6790,6792,6794,6796,6799,6805,6806,6807,6830,6831,6834,6835,6838,6854,6861,6862,6863,6869,6873,6885,6887,6890,6901,6902,6906,6908,6909,6914,6916,6918,6919,6921,6923,6924,6928,6931,6933,6934,6936,6937,6944,6949,6951,6952,6955,6957,6958,6962,6965,6969,6988,6994,7610,7631,7705,7711,7721,7722,7730,7732,7736,7740,7749,7750,7765,7769,7780,7786,7788,7791,7795,7799,7822,7823,8011,8016,8021,8028,8033,8039,8045,8046,8070,8072,8081,8101,8103,8104,8105,8110,8112,8114,8131,8150,8162,8163,8201,8210,8213,8215,8222,8249,8261,8271,8341,8367,8374,8404,8411,8422,8429,8438,8442,8443,8454,8462,8463,8464,8466,8467,8473,8476,8478,8481,8482,8487,8488,8499,8926,8940,8996,9103,910322,9105,910861,9110,911608,911622,911868,912000,9136,9802,9902,9904,9905,9906,9907,9908,9910,9911,9912,9914,9917,9918,9919,9921,9924,9925,9926,9927,9928,9929,9930,9931,9933,9934,9935,9937,9938,9939,9940,9941,9942,9943,9944,9945,9946,9955,9958"
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
