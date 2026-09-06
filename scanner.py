
from __future__ import annotations
import json, os, time
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO

import ccxt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

RSI_LEN = 14
EMA_LEN = 12
BOTTOM = 30.0
TOP = 70.0
RECLAIM = 60.0

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
REPORT_DIR = ROOT / "reports"
STATE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

SIGNAL_NAMES = {
    "STRONG": "💎 전환매수",
    "OVERSOLD": "🧊 과매도 진입",
    "OVERBOUGHT": "🔥 과매수 진입",
    "RECLAIM": "😮 분할매도",
}

def pine_rma(x: pd.Series, length: int) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.dropna()
    if len(valid) < length:
        return out
    seed_idx = valid.index[length - 1]
    seed_pos = x.index.get_loc(seed_idx)
    prev = valid.iloc[:length].mean()
    out.iloc[seed_pos] = prev
    alpha = 1.0 / length
    for i in range(seed_pos + 1, len(x)):
        val = x.iloc[i]
        if np.isnan(val):
            continue
        prev = alpha * val + (1 - alpha) * prev
        out.iloc[i] = prev
    return out

def pine_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    avg_up = pine_rma(up, length)
    avg_down = pine_rma(down, length)
    rs = avg_up / avg_down
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_down != 0, 100.0)
    rsi = rsi.where(avg_up != 0, 0.0)
    rsi = rsi.where(~((avg_up == 0) & (avg_down == 0)), 50.0)
    return rsi

def pine_ema(src: pd.Series, length: int) -> pd.Series:
    src = pd.to_numeric(src, errors="coerce")
    out = pd.Series(np.nan, index=src.index, dtype=float)
    alpha = 2.0 / (length + 1.0)
    prev = None
    for i, val in enumerate(src):
        if np.isnan(val):
            continue
        prev = float(val) if prev is None else alpha * float(val) + (1-alpha) * prev
        out.iloc[i] = prev
    return out

def co(a0, a1, b0, b1):
    return pd.notna(a0) and pd.notna(a1) and pd.notna(b0) and pd.notna(b1) and a0 > b0 and a1 <= b1

def cu(a0, a1, b0, b1):
    return pd.notna(a0) and pd.notna(a1) and pd.notna(b0) and pd.notna(b1) and a0 < b0 and a1 >= b1

def analyze(close: pd.Series):
    close = close.dropna().astype(float)
    if len(close) < 40:
        return None
    rsi = pine_rsi(close, RSI_LEN)
    sig = pine_ema(rsi, EMA_LEN)
    was_os = False
    t70 = False
    events = []

    for i in range(1, len(close)):
        r0, r1 = rsi.iloc[i], rsi.iloc[i-1]
        s0, s1 = sig.iloc[i], sig.iloc[i-1]
        if pd.isna(r0):
            continue
        if r0 < BOTTOM:
            was_os = True
        golden = co(r0, r1, s0, s1)
        strong = golden and was_os
        if golden:
            was_os = False
        igi = cu(r0, r1, BOTTOM, BOTTOM)
        eungdi = co(r0, r1, TOP, TOP)
        if r0 >= TOP:
            t70 = True
        haah = False
        if t70 and cu(r0, r1, RECLAIM, RECLAIM):
            haah = True
            t70 = False

        event = "STRONG" if strong else "OVERSOLD" if igi else "OVERBOUGHT" if eungdi else "RECLAIM" if haah else None
        if event:
            events.append({"date": pd.Timestamp(close.index[i]), "type": event, "rsi": float(r0), "close": float(close.iloc[i])})

    if not events:
        return None
    last = events[-1]
    latest = pd.Timestamp(close.index[-1])
    return {
        "eligible": last["type"] == "STRONG",
        "signal": SIGNAL_NAMES[last["type"]],
        "signal_code": last["type"],
        "signal_date": last["date"].date().isoformat(),
        "weeks_ago": max(0, int((latest - last["date"]).days // 7)),
        "current_rsi": round(float(rsi.dropna().iloc[-1]), 1),
        "signal_rsi": round(last["rsi"], 1),
        "close": last["close"] if len(close)==0 else float(close.iloc[-1]),
    }

def nasdaq_symbols():
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    symbols = set()
    for url in urls:
        txt = requests.get(url, headers=headers, timeout=30).text
        df = pd.read_csv(StringIO(txt), sep="|")
        sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper() != "Y"]
        for s in df[sym_col]:
            if pd.isna(s):
                continue
            s = str(s).strip()
            if not s or s.lower() == "nan" or "$" in s:
                continue
            symbols.add(s.replace(".", "-"))
    return sorted(symbols)

def drop_current_equity_week(s: pd.Series):
    # completed weekly bars only
    now = datetime.now(timezone.utc)
    if now.weekday() < 4 and len(s) > 1:
        return s.iloc[:-1]
    return s

def scan_us():
    tickers = nasdaq_symbols()
    results = []
    batch_size = 150
    print(f"[US] universe={len(tickers)}")
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start+batch_size]
        try:
            data = yf.download(
                tickers=batch, period="10y", interval="1wk", auto_adjust=True,
                group_by="ticker", threads=True, progress=False, timeout=40
            )
            if len(batch) == 1:
                data = {batch[0]: data["Close"].dropna()}
            else:
                temp = {}
                if isinstance(data.columns, pd.MultiIndex):
                    for t in batch:
                        try:
                            temp[t] = data[(t, "Close")].dropna()
                        except Exception:
                            pass
                data = temp
            for t, s in data.items():
                s = drop_current_equity_week(s)
                a = analyze(s)
                if a and a["eligible"]:
                    results.append({
                        "market":"US","symbol":t,"signal_date":a["signal_date"],
                        "weeks_ago":a["weeks_ago"],"current_rsi":a["current_rsi"],
                        "signal_rsi":a["signal_rsi"],"close":round(a["close"],4),
                        "chart":f"https://www.tradingview.com/chart/?symbol={t}",
                    })
        except Exception as e:
            print("[US batch error]", start, e)
        time.sleep(0.2)
    return pd.DataFrame(results)

def crypto_symbols(exchange_id="bybit", top_n=300):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    markets = ex.load_markets()
    stables = {"USDT","USDC","FDUSD","TUSD","DAI","USDE","USDS","PYUSD","EURC","USDP"}
    syms = [s for s,m in markets.items() if m.get("active",True) and m.get("spot") is True and m.get("quote")=="USDT" and m.get("base") not in stables]
    try:
        tickers = ex.fetch_tickers()
        ranked=[]
        for s in syms:
            t=tickers.get(s,{})
            q=t.get("quoteVolume")
            if q is None:
                q=(t.get("last") or 0)*(t.get("baseVolume") or 0)
            ranked.append((s,float(q or 0)))
        ranked.sort(key=lambda x:x[1], reverse=True)
        syms=[s for s,_ in ranked[:top_n]]
    except Exception:
        syms=syms[:top_n]
    return ex, syms

def scan_crypto(exchange_id="bybit", top_n=300):
    ex, syms = crypto_symbols(exchange_id, top_n)
    print(f"[CRYPTO] {exchange_id} universe={len(syms)}")
    results=[]
    for i,sym in enumerate(syms,1):
        try:
            candles=ex.fetch_ohlcv(sym,"1w",limit=500)
            if len(candles)<40: 
                continue
            df=pd.DataFrame(candles,columns=["ts","o","h","l","c","v"])
            df["date"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
            df=df.set_index("date")
            if len(df)>1:
                df=df.iloc[:-1]  # current week 제외
            a=analyze(df["c"])
            if a and a["eligible"]:
                base=sym.split("/")[0]
                ex_tv="BYBIT" if exchange_id=="bybit" else "BINANCE"
                results.append({
                    "market":exchange_id.upper(),"symbol":sym,"signal_date":a["signal_date"],
                    "weeks_ago":a["weeks_ago"],"current_rsi":a["current_rsi"],
                    "signal_rsi":a["signal_rsi"],"close":round(a["close"],8),
                    "chart":f"https://www.tradingview.com/chart/?symbol={ex_tv}:{base}USDT",
                })
        except Exception as e:
            print("[CRYPTO error]",sym,e)
    return pd.DataFrame(results)

def classify_new(df: pd.DataFrame, key: str, prev: set[str]):
    if df.empty:
        df["status"]=[]
        return df
    df["status"]=df[key].apply(lambda x: "🆕 신규" if x not in prev else "✅ 유지")
    return df

def load_prev(name):
    p=STATE_DIR/f"{name}.json"
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()

def save_state(name, vals):
    (STATE_DIR/f"{name}.json").write_text(json.dumps(sorted(vals),ensure_ascii=False,indent=2),encoding="utf-8")

def make_html(us: pd.DataFrame, cr: pd.DataFrame):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    def section(title, df):
        if df.empty:
            return f"<h2>{title}</h2><p>조건 충족 종목 없음</p>"
        d=df.copy()
        d["chart"]=d["chart"].apply(lambda u:f'<a href="{u}" target="_blank">차트</a>')
        return f"<h2>{title} ({len(d)})</h2>"+d.to_html(index=False,escape=False)
    html=f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>남씨표류기 Weekly</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#111;color:#eee;margin:20px}}
h1{{font-size:1.5rem}} h2{{margin-top:28px;font-size:1.15rem}}
table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}
th,td{{border:1px solid #333;padding:8px;white-space:nowrap}} th{{background:#222}}
a{{color:#6cf}} .note{{color:#aaa;font-size:13px}}
</style></head><body>
<h1>🏝️ 남씨표류기 — Weekly</h1>
<p class="note">생성: {now} / 조건: 가장 최근 주봉 신호가 💎 전환매수 / 미완성 금주 주봉 제외</p>
{section("🇺🇸 미국주식",us)}
{section("₿ 크립토",cr)}
</body></html>"""
    (REPORT_DIR/"latest.html").write_text(html,encoding="utf-8")
    return html

def send_telegram(us, cr):
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    chat_id=os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat_id:
        print("[Telegram] secrets not set; skipped.")
        return
    def fmt(title, df):
        lines=[title]
        if df.empty:
            lines.append("조건 충족 없음")
        else:
            for _,r in df.sort_values(["status","weeks_ago"]).head(25).iterrows():
                lines.append(f"{r['status']} {r['symbol']} | {r['signal_date']} | {r['weeks_ago']}주 | RSI {r['current_rsi']}")
            if len(df)>25:
                lines.append(f"...외 {len(df)-25}개")
        return "\n".join(lines)
    msg="🏝️ 남씨표류기 Weekly\n\n"+fmt("🇺🇸 미국주식",us)+"\n\n"+fmt("₿ 크립토",cr)
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id":chat_id,"text":msg,"disable_web_page_preview":True},
        timeout=20
    ).raise_for_status()

def main():
    prev_us=load_prev("us")
    prev_cr=load_prev("crypto")

    us=scan_us()
    cr=scan_crypto("bybit",300)

    us=classify_new(us,"symbol",prev_us)
    cr=classify_new(cr,"symbol",prev_cr)

    us=us.sort_values(["status","weeks_ago","signal_date"],ascending=[True,True,False]) if not us.empty else us
    cr=cr.sort_values(["status","weeks_ago","signal_date"],ascending=[True,True,False]) if not cr.empty else cr

    us.to_csv(REPORT_DIR/"us_latest.csv",index=False,encoding="utf-8-sig")
    cr.to_csv(REPORT_DIR/"crypto_latest.csv",index=False,encoding="utf-8-sig")
    make_html(us,cr)

    save_state("us",set(us["symbol"].tolist()) if not us.empty else set())
    save_state("crypto",set(cr["symbol"].tolist()) if not cr.empty else set())
    send_telegram(us,cr)
    print(f"Done. US={len(us)} Crypto={len(cr)}")

if __name__=="__main__":
    main()
