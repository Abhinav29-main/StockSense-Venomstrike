# StockSense - AI-Powered Market Intelligence

**Track: Claw & Shield** | OSSome Hacks 2026

---

## 📦 What's Inside?

- Random Forest ML predictions (63%-89% accuracy)
- FinBERT sentiment analysis on live news
- Technical indicators (RSI, MACD, SMA50, SMA200)
- Portfolio tracker with P&L
- OpenClaw agent interface
- ArmorIQ security shield

---

## 🔧 Setup

### 1. Install Python packages

```bash
pip install -r requirements.txt
2. Start the backend
bash
python stocksense_backend.py
Wait for: FinBERT ready. then Running on http://localhost:5050

3. Open the frontend
Double-click stocksense_frontend.html OR open it in your browser

🧪 Testing
✅ Test 1: Backend running
Open browser → Go to http://localhost:5050/api/health

Expected: {"status":"ok","finbert":"loaded"}

✅ Test 2: Analyze a stock
Go to http://localhost:5050/api/analyze/AAPL

Expected: JSON with ml_forecast.signal (BUY/SELL/HOLD)

✅ Test 3: OpenClaw + ArmorIQ
Go to http://localhost:5050/api/openclaw/NVDA

Expected: "armoriq": {"verified": true}

✅ Test 4: Frontend UI
Type AAPL in search bar

Click Analyze

See: ML signal + news sentiment + technicals

✅ Test 5: OpenClaw Agent Panel
Scroll to "🦞 OpenClaw Agent Interface"

Type Analyze AAPL

Click Send to Agent

ArmorIQ shows ✅ verified with intent token

✅ Test 6: Portfolio Tracker
Scroll to Portfolio Tracker

Add: Ticker MSFT, Shares 1, Buy Price 10

Click Add Position

P&L calculates automatically

📁 Project Files
File	What it does
stocksense_backend.py	Flask API + ML + FinBERT
stocksense_frontend.html	Dashboard UI
run_stocksense.py	One-click launcher
SKILL.md	OpenClaw skill definition
requirements.txt	Python dependencies
👥 Team Venomstrike
Satyam Garg (Lead)

Snehal Tiwari

Abhinav Sanjith

Ankit Rishiraaj

⚠️ Note
First search takes ~20 seconds (FinBERT loads once). No API keys needed.

