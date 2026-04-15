# Scan-Taiwan-Stock

使用 Streamlit 建立的台股 CCI 雙向轉折掃描工具，會搭配 Supertrend 趨勢條件過濾買點。

## 安裝

```bash
pip install -r requirements.txt
```

## 啟動

```bash
streamlit run app.py
```

## 功能

- 輸入多組台股代號進行掃描
- 以 CCI 黃金交叉搭配 Supertrend 多頭條件找出買點
- 顯示掃描結果並支援匯出 CSV
