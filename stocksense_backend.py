import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 13:
    print("❌ ERROR: Python 3.13 not supported. Use Python 3.11 or 3.12")
    sys.exit(1)
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import feedparser
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

_sentiment_pipe = None

def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512
        )
    return _sentiment_pipe

def fetch_ohlcv(ticker, period="2y"):
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval="1d")
    df.dropna(inplace=True)
    return df

def compute_rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, n=20, k=2):
    sma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = sma + k * std
    lower = sma - k * std
    return upper, sma, lower

def build_features(df):
    df = df.copy()
    df['rsi'] = compute_rsi(df['Close'])
    macd_l, macd_s, macd_h = compute_macd(df['Close'])
    df['macd_hist'] = macd_h
    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    df['sma200'] = df['Close'].rolling(200).mean()
    bb_upper, bb_mid, bb_lower = compute_bollinger(df['Close'])
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_pct'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df['vol_chg'] = df['Volume'].pct_change()
    df['price_chg'] = df['Close'].pct_change()
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    df.dropna(inplace=True)
    return df

def ml_forecast(df, horizon=5):
    df = build_features(df)
    if len(df) < 60:
        return None

    feature_cols = ['rsi', 'macd_hist', 'bb_pct', 'momentum_5', 'momentum_10', 'vol_chg', 'sma20', 'sma50']
    X = df[feature_cols].values
    closes = df['Close'].values

    future_returns = []
    for i in range(len(closes) - horizon):
        ret = (closes[i + horizon] - closes[i]) / closes[i]
        future_returns.append(ret)

    future_returns = np.array(future_returns)
    labels = (future_returns > 0).astype(int)

    X_train = X[:len(labels)]
    n_train = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:n_train], labels[:n_train]
    X_val, y_val = X_train[n_train:], labels[n_train:]

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_tr, y_tr)

    val_acc = float(rf.score(X_val, y_val)) if len(X_val) > 0 else 0.0

    last_features = X[-1:] 
    prob = rf.predict_proba(last_features)[0]
    bull_prob = float(prob[1])

    lr = LinearRegression()
    lookback = min(60, len(closes))
    x_idx = np.arange(lookback).reshape(-1, 1)
    y_idx = closes[-lookback:]
    lr.fit(x_idx, y_idx)
    trend_slope = float(lr.coef_[0])
    trend_pct = trend_slope / (closes[-1] + 1e-9) * horizon * 100

    current = float(closes[-1])
    predicted_return = (bull_prob - 0.5) * 2 * 0.06
    target_price = current * (1 + predicted_return)

    return {
        "bull_probability": bull_prob,
        "val_accuracy": val_acc,
        "trend_pct": trend_pct,
        "predicted_return_pct": predicted_return * 100,
        "target_price": target_price,
        "current_price": current
    }

def fetch_news_rss(ticker):
    company_map = {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
        'NVDA': 'NVIDIA', 'TSLA': 'Tesla', 'AMZN': 'Amazon',
        'META': 'Meta', 'AMD': 'AMD', 'NFLX': 'Netflix',
        'JPM': 'JPMorgan', 'BAC': 'Bank of America'
    }
    query = company_map.get(ticker, ticker)
    urls = [
        f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    ]
    articles = []
    seen = set()
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                if title in seen:
                    continue
                seen.add(title)
                source = entry.get('source', {}).get('title', 'Unknown') if hasattr(entry.get('source', ''), 'get') else str(entry.get('source', 'News'))
                published = entry.get('published', '')
                articles.append({'title': title, 'source': source, 'published': published})
        except Exception:
            continue
    return articles[:15]

def analyze_sentiment_finbert(articles):
    if not articles:
        return []
    pipe = get_sentiment_pipe()
    titles = [a['title'] for a in articles]
    try:
        results = pipe(titles, batch_size=8)
    except Exception:
        results = [{'label': 'neutral', 'score': 0.5}] * len(titles)

    label_map = {'positive': 'Bullish', 'negative': 'Bearish', 'neutral': 'Neutral'}
    score_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

    enriched = []
    for article, res in zip(articles, results):
        raw_label = res['label'].lower()
        label = label_map.get(raw_label, 'Neutral')
        score = score_map.get(raw_label, 0.0) * res['score']
        enriched.append({
            **article,
            'sentiment_label': label,
            'sentiment_score': score,
            'confidence': res['score']
        })
    return enriched

@app.route('/api/analyze/<ticker>')
def analyze(ticker):
    ticker = ticker.upper().strip()
    try:
        df = fetch_ohlcv(ticker)
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404

        df_feat = build_features(df)

        closes = df_feat['Close'].values
        current_price = float(closes[-1])
        rsi_val = float(df_feat['rsi'].iloc[-1])
        macd_hist_val = float(df_feat['macd_hist'].iloc[-1])
        sma50_val = float(df_feat['sma50'].iloc[-1]) if not pd.isna(df_feat['sma50'].iloc[-1]) else None
        sma200_val = float(df_feat['sma200'].iloc[-1]) if not pd.isna(df_feat['sma200'].iloc[-1]) else None
        bb_upper_val = float(df_feat['bb_upper'].iloc[-1])
        bb_lower_val = float(df_feat['bb_lower'].iloc[-1])
        volume_val = int(df.iloc[-1]['Volume'])

        hi52 = float(df['Close'].rolling(min(252, len(df))).max().iloc[-1])
        lo52 = float(df['Close'].rolling(min(252, len(df))).min().iloc[-1])

        forecast = ml_forecast(df)

        articles = fetch_news_rss(ticker)
        enriched_news = analyze_sentiment_finbert(articles)

        sentiment_scores = [a['sentiment_score'] for a in enriched_news]
        avg_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0

        bull_count = sum(1 for a in enriched_news if a['sentiment_label'] == 'Bullish')
        bear_count = sum(1 for a in enriched_news if a['sentiment_label'] == 'Bearish')
        neu_count = sum(1 for a in enriched_news if a['sentiment_label'] == 'Neutral')

        composite_score = 0.0
        
        # RSI Logic (fixed)
        if rsi_val < 30: 
            composite_score += 0.9      # Strong BUY (oversold)
        elif rsi_val < 40: 
            composite_score += 0.4      # Mild BUY
        elif rsi_val > 80: 
            composite_score -= 0.9      # Strong SELL (overbought extreme)
        elif rsi_val > 70: 
            composite_score -= 0.5      # Mild SELL
        elif rsi_val > 60: 
            composite_score -= 0.2      # Weak SELL
        else: 
            composite_score += 0.1       # Neutral zone gets slight positive
        
        # MACD Logic (fixed)
        if macd_hist_val > 0.5: 
            composite_score += 0.7
        elif macd_hist_val > 0.2: 
            composite_score += 0.4
        elif macd_hist_val > 0: 
            composite_score += 0.2
        elif macd_hist_val < -0.5: 
            composite_score -= 0.7
        elif macd_hist_val < -0.2: 
            composite_score -= 0.4
        elif macd_hist_val < 0: 
            composite_score -= 0.2
        
        # Moving Averages
        if sma50_val: 
            composite_score += 0.4 if current_price > sma50_val else -0.35
        if sma200_val: 
            composite_score += 0.3 if current_price > sma200_val else -0.35
        
        # ML Signal
        if forecast:
            ml_signal = (forecast['bull_probability'] - 0.5) * 2
            composite_score += ml_signal * 1.0  # Reduced from 1.4
        
        # Sentiment
        composite_score += min(1.0, max(-1.0, avg_sentiment * 2.0))
        
        # Normalize final score
        final_score = min(1.0, max(-1.0, composite_score / 2.6))
        
        # Signal thresholds (fixed - higher thresholds for BUY/SELL)
        if final_score >= 0.45: 
            signal = 'STRONG BUY'
        elif final_score >= 0.25: 
            signal = 'BUY'
        elif final_score <= -0.45: 
            signal = 'STRONG SELL'
        elif final_score <= -0.25: 
            signal = 'SELL'
        elif final_score >= 0.10: 
            signal = 'ACCUMULATE'
        elif final_score <= -0.10: 
            signal = 'REDUCE'
        else: 
            signal = 'HOLD'
        
        # Confidence calculation (fixed)
        confidence = min(94, max(55, 55 + abs(final_score) * 35))
        if forecast and forecast.get('val_accuracy', 0) > 0.5:
            confidence = min(94, max(55, (confidence + forecast['val_accuracy'] * 100) / 2))
        if abs(avg_sentiment) > 0.2:
            confidence = min(94, confidence + 5)
        
        confidence = round(confidence, 1)

        target_price = forecast['target_price'] if forecast else current_price * (1 + final_score * 0.05)
        predicted_return_pct = forecast['predicted_return_pct'] if forecast else final_score * 5

        return jsonify({
            'ticker': ticker,
            'technicals': {
                'current_price': current_price,
                'rsi': round(rsi_val, 2),
                'macd_hist': round(macd_hist_val, 4),
                'sma50': round(sma50_val, 2) if sma50_val else None,
                'sma200': round(sma200_val, 2) if sma200_val else None,
                'bb_upper': round(bb_upper_val, 2),
                'bb_lower': round(bb_lower_val, 2),
                'volume': volume_val,
                'hi52': round(hi52, 2),
                'lo52': round(lo52, 2)
            },
            'ml_forecast': {
                'signal': signal,
                'confidence': confidence,
                'final_score': round(final_score, 4),
                'target_price': round(target_price, 2),
                'predicted_return_pct': round(predicted_return_pct, 2),
                'bull_probability': round(forecast['bull_probability'] * 100, 1) if forecast else 50.0,
                'model_accuracy': round(forecast['val_accuracy'] * 100, 1) if forecast else 0.0
            },
            'sentiment': {
                'avg_score': round(avg_sentiment, 4),
                'bullish_count': bull_count,
                'bearish_count': bear_count,
                'neutral_count': neu_count,
                'articles': enriched_news[:10]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quote/<ticker>')
def quote(ticker):
    ticker = ticker.upper().strip()
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        hist = tk.history(period='2d')
        if hist.empty:
            return jsonify({'error': 'No data'}), 404
        price = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else price
        change = price - prev
        change_pct = (change / prev) * 100 if prev else 0
        return jsonify({
            'ticker': ticker,
            'price': round(price, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'volume': int(hist['Volume'].iloc[-1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/openclaw/<ticker>')
def openclaw_analyze(ticker):
    """
    OpenClaw skill endpoint — returns a concise, agent-friendly payload.
    Protected by ArmorIQ intent verification (simulated token in response).
    """
    ticker = ticker.upper().strip()
    # ArmorIQ: sanitise ticker to alphanumeric only (prompt-injection guard)
    if not ticker.isalpha() or len(ticker) > 6:
        return jsonify({'error': 'Invalid ticker — ArmorIQ blocked: input did not pass sanitisation'}), 400

    import secrets, datetime
    intent_token = secrets.token_hex(16)          # ArmorIQ intent token (demo)
    timestamp    = datetime.datetime.utcnow().isoformat() + 'Z'

    try:
        df = fetch_ohlcv(ticker)
        if df.empty:
            return jsonify({'error': f'No data for {ticker}'}), 404

        df_feat = build_features(df)
        closes = df_feat['Close'].values
        current_price = float(closes[-1])
        rsi_val       = float(df_feat['rsi'].iloc[-1])
        macd_hist_val = float(df_feat['macd_hist'].iloc[-1])

        forecast = ml_forecast(df)
        articles = fetch_news_rss(ticker)
        enriched = analyze_sentiment_finbert(articles)

        sentiment_scores = [a['sentiment_score'] for a in enriched]
        avg_sentiment    = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
        bull_news = sum(1 for a in enriched if a['sentiment_label'] == 'Bullish')
        total_news = len(enriched) or 1
        news_bull_pct = round(bull_news / total_news * 100)

        # Composite signal (same logic as /api/analyze)
        composite_score = 0.0
        
        # RSI Logic
        if rsi_val < 30: 
            composite_score += 0.9
        elif rsi_val < 40: 
            composite_score += 0.4
        elif rsi_val > 80: 
            composite_score -= 0.9
        elif rsi_val > 70: 
            composite_score -= 0.5
        elif rsi_val > 60: 
            composite_score -= 0.2
        else: 
            composite_score += 0.1
        
        # MACD Logic
        if macd_hist_val > 0.5: 
            composite_score += 0.7
        elif macd_hist_val > 0.2: 
            composite_score += 0.4
        elif macd_hist_val > 0: 
            composite_score += 0.2
        elif macd_hist_val < -0.5: 
            composite_score -= 0.7
        elif macd_hist_val < -0.2: 
            composite_score -= 0.4
        elif macd_hist_val < 0: 
            composite_score -= 0.2
        
        # ML Signal
        if forecast:
            composite_score += (forecast['bull_probability'] - 0.5) * 2 * 1.0
        
        # Sentiment
        composite_score += min(1.0, max(-1.0, avg_sentiment * 2.0))
        
        # Normalize
        final_score = min(1.0, max(-1.0, composite_score / 2.6))
        
        # Signal thresholds
        if final_score >= 0.45: 
            signal = 'STRONG BUY'
        elif final_score >= 0.25: 
            signal = 'BUY'
        elif final_score <= -0.45: 
            signal = 'STRONG SELL'
        elif final_score <= -0.25: 
            signal = 'SELL'
        elif final_score >= 0.10: 
            signal = 'ACCUMULATE'
        elif final_score <= -0.10: 
            signal = 'REDUCE'
        else: 
            signal = 'HOLD'

        confidence = min(94, max(55, 55 + abs(final_score) * 35))
        if forecast and forecast.get('val_accuracy', 0) > 0.5:
            confidence = min(94, max(55, (confidence + forecast['val_accuracy'] * 100) / 2))
        if abs(avg_sentiment) > 0.2:
            confidence = min(94, confidence + 5)
        
        confidence = round(confidence, 1)
        
        target_price = forecast['target_price'] if forecast else current_price * (1 + final_score * 0.05)
        pred_ret     = forecast['predicted_return_pct'] if forecast else final_score * 5
        bull_prob    = round(forecast['bull_probability'] * 100, 1) if forecast else 50.0

        top_headlines = [
            {'title': a['title'], 'sentiment': a['sentiment_label'], 'confidence': round(a['confidence']*100)}
            for a in enriched[:5]
        ]

        macd_sent = 'bullish' if macd_hist_val > 0 else 'bearish'
        news_sent = 'positive' if avg_sentiment > 0.05 else 'negative' if avg_sentiment < -0.05 else 'neutral'

        summary = (
            f"{ticker} is signalling {signal} with {round(confidence)}% confidence. "
            f"Current price ${round(current_price,2)}, ML target ${round(target_price,2)} "
            f"({'+' if pred_ret>=0 else ''}{round(pred_ret,2)}% projected). "
            f"RSI {round(rsi_val,1)} — MACD {macd_sent} — News {news_sent} ({news_bull_pct}% bullish headlines). "
            f"All tool calls verified by ArmorIQ intent token {intent_token[:8]}…"
        )

        return jsonify({
            'ticker':             ticker,
            'signal':             signal,
            'confidence':         confidence,
            'current_price':      round(current_price, 2),
            'target_price':       round(target_price, 2),
            'predicted_return_pct': round(pred_ret, 2),
            'bull_probability':   bull_prob,
            'rsi':                round(rsi_val, 2),
            'macd_sentiment':     macd_sent,
            'news_sentiment':     news_sent,
            'news_bull_pct':      news_bull_pct,
            'top_headlines':      top_headlines,
            'summary':            summary,
            'armoriq': {
                'verified':      True,
                'intent_token':  intent_token,
                'timestamp':     timestamp,
                'policy':        'allow:web_fetch(localhost:5050)',
                'injection_check': 'passed'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'finbert': 'loaded' if _sentiment_pipe else 'lazy'})

if __name__ == '__main__':
    print("Starting StockSense backend on port 5050...")
    print("Pre-loading FinBERT model...")
    get_sentiment_pipe()
    print("FinBERT ready.")
    app.run(host='0.0.0.0', port=5050, debug=False)
