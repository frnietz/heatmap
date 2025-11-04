# Streamlit Stock Heatmap — Pro

**New features**
- Tabs: **Heatmap**, **Sectors**, **Watchlist**, **Alerts**
- **Return window** selector (1D..1Y or Custom)
- **Size by market cap**, **color by return**
- Watchlist persisted to `store.json`
- Simple **daily % change alerts** (manual refresh; webhook/email placeholder for production)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Production ideas
- Replace `yfinance` with your data provider.
- Add auto-refresh (e.g., every 5 minutes) behind a toggle.
- Implement email/webhook alerts (Slack/Telegram) in place of the manual "Check Alerts".
