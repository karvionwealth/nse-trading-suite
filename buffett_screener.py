import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz

# ---------- CONFIG ----------
# Reads from GitHub Secrets
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')

# --- Support for multiple chat IDs ---
CHAT_IDS_MULTI = os.environ.get('TELEGRAM_CHAT_IDS')

if CHAT_IDS_MULTI:
    CHAT_IDS = [cid.strip() for cid in CHAT_IDS_MULTI.split(',') if cid.strip()]
else:
    single_id = os.environ.get('TELEGRAM_CHAT_ID')
    CHAT_IDS = [single_id] if single_id else []
# ------------------------------------------

# 128 liquid NSE stocks
SYMBOLS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
    "BAJFINANCE.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","NTPC.NS","ONGC.NS",
    "POWERGRID.NS","WIPRO.NS","HCLTECH.NS","ULTRACEMCO.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","ADANIPORTS.NS","ADANIENT.NS","DIVISLAB.NS","DRREDDY.NS",
    "CIPLA.NS","BRITANNIA.NS","HDFCLIFE.NS","SBILIFE.NS","EICHERMOT.NS","M&M.NS",
    "HINDZINC.NS","VEDL.NS","DLF.NS","INDIGO.NS","HAVELLS.NS","VOLTAS.NS",
    "DABUR.NS","PIDILITIND.NS","BERGEPAINT.NS","LUPIN.NS","AUROPHARMA.NS",
    "BIOCON.NS","TORNTPHARM.NS","ALKEM.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "BAJAJFINSV.NS","BAJAJHLDNG.NS","BALKRISIND.NS","BANDHANBNK.NS","BEL.NS",
    "BHARATFORG.NS","BOSCHLTD.NS","BPCL.NS","CANBK.NS","CHOLAFIN.NS","COALINDIA.NS",
    "COLPAL.NS","CONCOR.NS","CUMMINSIND.NS","DEEPAKNTR.NS","ESCORTS.NS",
    "GAIL.NS","GODREJCP.NS","GODREJPROP.NS","GRASIM.NS","HAL.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDPETRO.NS","ICICIPRULI.NS",
    "IDFCFIRSTB.NS","INDUSINDBK.NS","INDUSTOWER.NS","IOC.NS","IRCTC.NS",
    "JINDALSTEL.NS","JUBLFOOD.NS","LICHSGFIN.NS","M&MFIN.NS","MARICO.NS",
    "MFSL.NS","MOTHERSON.NS","MPHASIS.NS","MRF.NS","MUTHOOTFIN.NS","NAUKRI.NS",
    "NAVINFLUOR.NS","NESTLEIND.NS","OBEROIRLTY.NS","OFSS.NS","PAGEIND.NS",
    "PERSISTENT.NS","PETRONET.NS","PFC.NS","PIIND.NS","PNB.NS","POLYCAB.NS",
    "POONAWALLA.NS","PRESTIGE.NS","RAMCOCEM.NS","RBLBANK.NS","RECLTD.NS",
    "SAIL.NS","SBICARD.NS","SHREECEM.NS","SIEMENS.NS","SRF.NS","SUNTV.NS",
    "SYNGENE.NS","TATACHEM.NS","TATACOMM.NS","TATACONSUM.NS","TECHM.NS",
    "TIINDIA.NS","TRENT.NS","TVSMOTOR.NS","UPL.NS","YESBANK.NS","ZEEL.NS",
    "PAYTM.NS","POLICYBZR.NS","NYKAA.NS","DELHIVERY.NS"
]

def send_telegram(msg):
    """Send message to multiple Telegram recipients."""
    if not TELEGRAM_TOKEN or not CHAT_IDS:
        print("❌ ERROR: TELEGRAM_TOKEN or CHAT_IDS not set.")
        return
    
    for chat_id in CHAT_IDS:
        if not chat_id:
            continue
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            response = requests.post(
                url, 
                json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, 
                timeout=10
            )
            response.raise_for_status()
            print(f"✅ Telegram sent to {chat_id}")
        except Exception as e:
            print(f"❌ Telegram error for {chat_id}: {e}")

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    df.columns = [str(col).strip() for col in df.columns]
    if len(df.columns) >= 4:
        if 'Close' not in df.columns:
            df['Close'] = df.iloc[:, 3]
        if 'High' not in df.columns and len(df.columns) >= 2:
            df['High'] = df.iloc[:, 1]
        if 'Low' not in df.columns and len(df.columns) >= 3:
            df['Low'] = df.iloc[:, 2]
        if 'Volume' not in df.columns and len(df.columns) >= 5:
            df['Volume'] = df.iloc[:, 4] if len(df.columns) > 4 else df.iloc[:, 5]
    for col in df.columns:
        if 'close' in col.lower() and 'Close' not in df.columns:
            df['Close'] = df[col]
        if 'volume' in col.lower() and 'Volume' not in df.columns:
            df['Volume'] = df[col]
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_fundamentals(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        roe = info.get('returnOnEquity', 0)
        if roe is None: roe = 0
        debt = info.get('debtToEquity', 100)
        if debt is None: debt = 100
        pe = info.get('trailingPE', 100)
        if pe is None: pe = 100
        return roe, debt, pe
    except:
        return 0, 100, 100

def get_technicals(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        if len(df) < 200:
            return None, None, None
        df = normalize_columns(df)
        close = float(df['Close'].iloc[-1])
        sma200 = float(df['Close'].rolling(200).mean().iloc[-1])
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])
        return close, sma200, rsi_val
    except:
        return None, None, None

def generate_buffett_picks():
    """
    Strict Buffett Filter:
    - ROE > 15%
    - Debt/Equity < 0.5
    - P/E < 30
    - Price < 200-day SMA (trading at a discount)
    - RSI < 45 (oversold)
    """
    picks = []
    for sym in SYMBOLS:
        roe, debt, pe = get_fundamentals(sym)
        if roe < 15 or debt > 0.5 or pe > 30 or pe < 0:
            continue
        
        close, sma200, rsi = get_technicals(sym)
        if close is None:
            continue
        
        if close < sma200 and rsi < 45:
            discount_pct = round((1 - close / sma200) * 100, 2)
            picks.append({
                'Stock': sym.replace('.NS',''),
                'Price': round(close, 2),
                'SMA 200': round(sma200, 2),
                'Discount %': discount_pct,
                'RSI': round(rsi, 2),
                'ROE': round(roe, 2),
                'P/E': round(pe, 2)
            })
    return picks

def send_report():
    """Generate and send Buffett report to Telegram."""
    send_telegram("🧠 BUFFETT SCREENER INITIALIZED. Scanning for value picks...")

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    date_str = now.strftime("%d %B %Y, %I:%M %p")

    print(f"🔄 Running Buffett Screener for {date_str}...")
    picks = generate_buffett_picks()
    print(f"✅ Found {len(picks)} value picks.")

    if picks:
        message = f"🧠 <b>WARREN BUFFETT VALUE PICKS</b>\n📅 {date_str}\n━━━━━━━━━━━━━━━\n\n"
        message += "💡 <i>Strict Filters: ROE>15%, Debt<0.5, P/E<30, Price < 200 SMA, RSI<45</i>\n\n"
        
        for p in picks[:10]:  # Top 10
            message += f"📈 <b>{p['Stock']}</b>\n"
            message += f"   Price: ₹{p['Price']}\n"
            message += f"   Discount to 200 SMA: <b>{p['Discount %']}%</b>\n"
            message += f"   RSI: {p['RSI']} | ROE: {p['ROE']}% | P/E: {p['P/E']}\n\n"
        
        message += "━━━━━━━━━━━━━━━\n"
        message += f"✅ Total Picks Found: {len(picks)}\n"
        message += "⚠️ Strategy: Buy on dips. Hold for 6-12 months. No SL."
    else:
        message = f"🧠 <b>WARREN BUFFETT VALUE PICKS</b>\n📅 {date_str}\n━━━━━━━━━━━━━━━\n\n"
        message += "❌ No stocks passed the strict Buffett filters today.\n"
        message += "💡 In Buffett's words: <i>'Be greedy when others are fearful.'</i>\n"
        message += "   Wait for a market correction to buy quality at a discount."

    print("📤 Sending Buffett report...")
    send_telegram(message)
    print("✅ Buffett report complete!")

if __name__ == "__main__":
    print("🚀 Starting Warren Buffett Screener...")
    send_report()
