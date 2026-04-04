import requests
from shield import ArmorIQShield

# Simulated mode – no real Alpaca keys needed
SIMULATE_TRADES = True   # Set to False if you later get keys

FIXED_TRADE_AMOUNT = 100.0

def run_agent(ticker):
    print(f"\n🦞 OpenClaw Agent – {ticker}")
    
    # 1. Reasoning
    try:
        resp = requests.get(f"http://localhost:5050/api/openclaw/{ticker}", timeout=10)
        data = resp.json()
        signal = data.get("signal")
        confidence = data.get("confidence", 0)
        print(f"   Signal: {signal}, Confidence: {confidence}%")
    except Exception as e:
        print(f"   ❌ Backend error: {e}")
        return

    if signal == "SELL":
       print("   📉 SELL signal – would short or sell position (not implemented).")
       return
    elif signal != "BUY":
       print(f"   ⏸️ Signal {signal} not actionable. Halted.")
       return

    # 2. ArmorIQ enforcement
    shield = ArmorIQShield()
    ok, reason = shield.validate_intent(ticker, FIXED_TRADE_AMOUNT, confidence)
    print(f"   🛡️ Shield: {reason}")
    if not ok:
        print("   🚫 Trade BLOCKED by ArmorIQ.")
        return

    # 3. Execute trade (simulated or real)
    print("   ✅ Shield passed. Executing trade...")
    if SIMULATE_TRADES:
        print(f"   📈 [SIMULATED] Market order: BUY ${FIXED_TRADE_AMOUNT} of {ticker}")
        print("   (With Alpaca keys, this would execute a real paper trade.)")
    else:
        # Real execution code (requires valid keys)
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            # 🔐 Replace with your keys when available
            ALPACA_API_KEY = "PK..."
            ALPACA_SECRET_KEY = "SK..."

            client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            quote = client.get_quote(ticker)
            price = quote.bid_price or quote.ask_price
            shares = FIXED_TRADE_AMOUNT / price
            order = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            result = client.submit_order(order)
            print(f"   📈 REAL Order placed: {result.id} at ~${price:.2f}")
        except ImportError:
            print("   ⚠️ alpaca-py not installed. Run: pip install alpaca-py")
        except Exception as e:
            print(f"   ❌ Trade error: {e}")

if __name__ == "__main__":
    # Demo A – passes shield (if confidence >= policy threshold)
    run_agent("AAPL")
    run_agent("TSLA")
    run_agent("NVDA")
    run_agent("MSFT")
    # Demo B – blocked by restricted ticker (if GME in policy.json)
    run_agent("GME")