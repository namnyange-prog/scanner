from __future__ import annotations

import json
import os
import time
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

RESULT_COLUMNS = [
    "market", "symbol", "status", "signal_date", "weeks_ago",
    "current_rsi", "signal_rsi", "close", "chart"
]


def empty_results():
    return pd.DataFrame(columns=RESULT_COLUMNS)


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
        prev = float(val) if prev is None else alpha * float(val) + (1 - alpha) * prev
        out.iloc[i] = prev
    return out


def co(a0, a1, b0, b1):
    return (
        pd.notna(a0) and pd.notna(a1) and pd.notna(b0) and pd.notna(b1)
        and a0 > b0 and a1 <= b1
    )


def cu(a0, a1, b0, b1):
    return (
        pd.notna(a0) and pd.notna(a1) and pd.notna(b0) and pd.notna(b1)
        and a0 < b0 and a1 >= b1
    )


def analyze(close: pd.Series):
    close = close.dropna().astype(float)
    if len(close) < 40:
        return None

    rsi = pine_rsi(close, RSI_LEN)
    sig = pine_ema(rsi, EMA_LEN)
    was_os = False
    touched70 = False
    events = []

    for i in range(1, len(close)):
        r0, r1 = rsi.iloc[i], rsi.iloc[i - 1]
        s0, s1 = sig.iloc[i], sig.iloc[i - 1]

        if pd.isna(r0):
            continue

        if r0 < BOTTOM:
            was_os = True

        golden = co(r0, r1, s0, s1)
        strong = golden and was_os
        if golden:
            was_os = False

        oversold = cu(r0, r1, BOTTOM, BOTTOM)
        overbought = co(r0, r1, TOP, TOP)

        if r0 >= TOP:
            touched70 = True

        reclaim = False
        if touched70 and cu(r0, r1, RECLAIM, RECLAIM):
            reclaim = True
            touched70 = False

        event = (
            "STRONG" if strong else
            "OVERSOLD" if oversold else
            "OVERBOUGHT" if overbought else
            "RECLAIM" if reclaim else
            None
        )

        if event:
            events.append({
                "date": pd.Timestamp(close.index[i]),
                "type": event,
                "rsi": float(r0),
            })

    if not events or rsi.dropna().empty:
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
        "close": float(close.iloc[-1]),
    }


# -----------------------------
# 미국 주식
# -----------------------------

def nasdaq_symbols():
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    symbols = set()

    for url in urls:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), sep="|")

        sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"

        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper() != "Y"]

        for raw in df[sym_col]:
            if pd.isna(raw):
                continue

            s = str(raw).strip().upper()

            if (
                not s
                or s == "NAN"
                or "$" in s
                or "FILE CREATION TIME" in s
                or len(s) > 10
            ):
                continue

            # Yahoo 형식
            s = s.replace(".", "-")
            symbols.add(s)

    return sorted(symbols)


def drop_current_equity_week(s: pd.Series):
    # 월~목에는 Yahoo의 현재 미완성 주봉이 섞일 수 있으므로 마지막 봉 제거
    now = datetime.now(timezone.utc)
    if now.weekday() < 4 and len(s) > 1:
        return s.iloc[:-1]
    return s


def extract_yahoo_close(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    out = {}

    if data is None or data.empty:
        return out

    if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns:
            s = data["Close"].dropna()
            if not s.empty:
                out[tickers[0]] = s
        return out

    if not isinstance(data.columns, pd.MultiIndex):
        return out

    # yfinance 버전에 따라 (ticker, field) 또는 (field, ticker) 둘 다 대응
    lv0 = set(map(str, data.columns.get_level_values(0)))
    lv1 = set(map(str, data.columns.get_level_values(1)))

    for t in tickers:
        candidates = []
        if t in lv0:
            candidates.append((t, "Close"))
        if t in lv1:
            candidates.append(("Close", t))

        for key in candidates:
            try:
                s = data[key].dropna()
                if not s.empty:
                    out[t] = s
                    break
            except Exception:
                pass

    return out


def yahoo_download_batch(batch: list[str], attempts: int = 3):
    remaining = list(batch)
    found = {}

    for attempt in range(1, attempts + 1):
        if not remaining:
            break

        try:
            data = yf.download(
                tickers=remaining,
                period="10y",
                interval="1wk",
                auto_adjust=True,
                group_by="ticker",
                threads=False,          # 동시 요청 폭주 방지
                progress=False,
                timeout=45,
            )
            got = extract_yahoo_close(data, remaining)
            found.update(got)
            remaining = [t for t in remaining if t not in got]
        except Exception as e:
            print(f"[US retry {attempt}/{attempts}] batch exception: {e}")

        if remaining and attempt < attempts:
            wait = 5 * attempt
            print(f"[US retry] {len(remaining)} tickers missing -> wait {wait}s")
            time.sleep(wait)

    return found, remaining


def scan_us():
    tickers = nasdaq_symbols()
    results = []
    failed = []
    batch_size = 60

    print(f"[US] universe={len(tickers)}")

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        batch_no = start // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        print(f"[US] batch {batch_no}/{total_batches} ({len(batch)} tickers)")

        found, missing = yahoo_download_batch(batch, attempts=3)
        failed.extend(missing)

        for t, s in found.items():
            try:
                s = drop_current_equity_week(s)
                a = analyze(s)

                if a and a["eligible"]:
                    results.append({
                        "market": "US",
                        "symbol": t,
                        "signal_date": a["signal_date"],
                        "weeks_ago": a["weeks_ago"],
                        "current_rsi": a["current_rsi"],
                        "signal_rsi": a["signal_rsi"],
                        "close": round(a["close"], 4),
                        "chart": f"https://www.tradingview.com/chart/?symbol={t}",
                    })
            except Exception as e:
                print(f"[US analyze error] {t}: {e}")

        # Yahoo rate limit 완화
        time.sleep(1.0)

    scanned = len(tickers) - len(failed)
    coverage = scanned / len(tickers) if tickers else 0

    print(
        f"[US SUMMARY] total={len(tickers)} scanned={scanned} "
        f"failed={len(failed)} coverage={coverage:.1%} eligible={len(results)}"
    )

    if failed:
        print("[US FAILED SAMPLE]", ", ".join(failed[:30]))

    df = pd.DataFrame(results)
    if df.empty:
        df = empty_results().drop(columns=["status"])

    return df, {
        "source": "Yahoo Finance",
        "total": len(tickers),
        "scanned": scanned,
        "failed": len(failed),
        "coverage": coverage,
        "ok": coverage >= 0.70,
    }


# -----------------------------
# 크립토
# Bybit GitHub IP 차단 대비:
# Binance -> OKX -> Kraken 순서 자동 대체
# -----------------------------

CRYPTO_EXCHANGES = ["binance", "okx", "kraken"]

TV_EXCHANGE = {
    "binance": "BINANCE",
    "okx": "OKX",
    "kraken": "KRAKEN",
}


def make_exchange(exchange_id: str):
    cls = getattr(ccxt, exchange_id)
    ex = cls({
        "enableRateLimit": True,
        "timeout": 30000,
    })
    return ex


def choose_crypto_exchange():
    last_error = None

    for exchange_id in CRYPTO_EXCHANGES:
        try:
            print(f"[CRYPTO] trying {exchange_id}...")
            ex = make_exchange(exchange_id)
            markets = ex.load_markets()

            # 최소한 시장정보를 정상적으로 받았는지 확인
            if not markets:
                raise RuntimeError("empty markets")

            print(f"[CRYPTO] selected exchange={exchange_id}")
            return ex, exchange_id, markets

        except Exception as e:
            last_error = e
            print(f"[CRYPTO] {exchange_id} unavailable: {type(e).__name__}: {e}")

    raise RuntimeError(f"All crypto exchanges unavailable. last_error={last_error}")


def crypto_symbols(ex, exchange_id: str, markets: dict, top_n=300):
    stables = {
        "USDT", "USDC", "FDUSD", "TUSD", "DAI",
        "USDE", "USDS", "PYUSD", "EURC", "USDP"
    }

    quote_priority = ["USDT", "USD"]

    syms = []
    for quote in quote_priority:
        syms = [
            s for s, m in markets.items()
            if m.get("active", True)
            and m.get("spot") is True
            and m.get("quote") == quote
            and m.get("base") not in stables
        ]
        if len(syms) >= 50:
            break

    # 거래대금 순 정렬 시도. 실패하면 시장목록 순서 사용.
    try:
        tickers = ex.fetch_tickers()
        ranked = []

        for s in syms:
            t = tickers.get(s, {})
            q = t.get("quoteVolume")
            if q is None:
                q = (t.get("last") or 0) * (t.get("baseVolume") or 0)

            try:
                q = float(q or 0)
            except Exception:
                q = 0

            ranked.append((s, q))

        ranked.sort(key=lambda x: x[1], reverse=True)
        syms = [s for s, _ in ranked[:top_n]]

    except Exception as e:
        print(f"[CRYPTO] fetch_tickers unavailable: {e}")
        syms = syms[:top_n]

    return syms


def scan_crypto(top_n=300):
    try:
        ex, exchange_id, markets = choose_crypto_exchange()
        syms = crypto_symbols(ex, exchange_id, markets, top_n)
    except Exception as e:
        print(f"[CRYPTO FATAL] {e}")
        return empty_results().drop(columns=["status"]), {
            "source": "none",
            "total": 0,
            "scanned": 0,
            "failed": 0,
            "coverage": 0.0,
            "ok": False,
        }

    print(f"[CRYPTO] {exchange_id} universe={len(syms)}")

    results = []
    failed = []

    for i, sym in enumerate(syms, 1):
        if i == 1 or i % 25 == 0:
            print(f"[CRYPTO] {i}/{len(syms)}")

        success = False

        for attempt in range(1, 4):
            try:
                candles = ex.fetch_ohlcv(sym, "1w", limit=500)

                if len(candles) < 40:
                    success = True
                    break

                df = pd.DataFrame(
                    candles,
                    columns=["ts", "o", "h", "l", "c", "v"]
                )
                df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df = df.set_index("date")

                # 현재 진행중 주봉 제거
                if len(df) > 1:
                    df = df.iloc[:-1]

                a = analyze(df["c"])

                if a and a["eligible"]:
                    base = str(markets.get(sym, {}).get("base") or sym.split("/")[0])
                    quote = str(markets.get(sym, {}).get("quote") or sym.split("/")[-1])
                    tv_ex = TV_EXCHANGE.get(exchange_id, exchange_id.upper())
                    tv_symbol = f"{base}{quote}".replace("-", "").replace("/", "")

                    results.append({
                        "market": exchange_id.upper(),
                        "symbol": sym,
                        "signal_date": a["signal_date"],
                        "weeks_ago": a["weeks_ago"],
                        "current_rsi": a["current_rsi"],
                        "signal_rsi": a["signal_rsi"],
                        "close": round(a["close"], 8),
                        "chart": (
                            "https://www.tradingview.com/chart/"
                            f"?symbol={tv_ex}:{tv_symbol}"
                        ),
                    })

                success = True
                break

            except Exception as e:
                if attempt < 3:
                    wait = 2 * attempt
                    print(
                        f"[CRYPTO retry] {sym} {attempt}/3 "
                        f"{type(e).__name__} -> {wait}s"
                    )
                    time.sleep(wait)
                else:
                    print(f"[CRYPTO error] {sym}: {type(e).__name__}: {e}")

        if not success:
            failed.append(sym)

    scanned = len(syms) - len(failed)
    coverage = scanned / len(syms) if syms else 0

    print(
        f"[CRYPTO SUMMARY] source={exchange_id} total={len(syms)} "
        f"scanned={scanned} failed={len(failed)} "
        f"coverage={coverage:.1%} eligible={len(results)}"
    )

    if failed:
        print("[CRYPTO FAILED SAMPLE]", ", ".join(failed[:30]))

    df = pd.DataFrame(results)
    if df.empty:
        df = empty_results().drop(columns=["status"])

    return df, {
        "source": exchange_id,
        "total": len(syms),
        "scanned": scanned,
        "failed": len(failed),
        "coverage": coverage,
        "ok": coverage >= 0.70,
    }


def classify_new(df: pd.DataFrame, key: str, prev: set[str]):
    df = df.copy()
    if "status" not in df.columns:
        df["status"] = pd.Series(dtype="object")

    if df.empty:
        return df

    df["status"] = df[key].apply(
        lambda x: "🆕 신규" if x not in prev else "✅ 유지"
    )
    return df


def load_prev(name):
    p = STATE_DIR / f"{name}.json"

    if not p.exists():
        return set()

    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_state(name, vals):
    (STATE_DIR / f"{name}.json").write_text(
        json.dumps(sorted(vals), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_html(us: pd.DataFrame, cr: pd.DataFrame, us_meta: dict, cr_meta: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def section(title, df):
        if df.empty:
            return f"<h2>{title}</h2><p>조건 충족 종목 없음</p>"

        d = df.copy()
        d["chart"] = d["chart"].apply(
            lambda u: f'<a href="{u}" target="_blank">차트</a>'
        )
        return (
            f"<h2>{title} ({len(d)})</h2>"
            + d.to_html(index=False, escape=False)
        )

    us_status = (
        f"{us_meta['scanned']}/{us_meta['total']} "
        f"({us_meta['coverage']:.1%}) / 실패 {us_meta['failed']}"
    )
    cr_status = (
        f"{cr_meta['source']} / {cr_meta['scanned']}/{cr_meta['total']} "
        f"({cr_meta['coverage']:.1%}) / 실패 {cr_meta['failed']}"
    )

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>남씨표류기 Weekly</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#111;color:#eee;margin:20px}}
h1{{font-size:1.5rem}}
h2{{margin-top:28px;font-size:1.15rem}}
table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}
th,td{{border:1px solid #333;padding:8px;white-space:nowrap}}
th{{background:#222}}
a{{color:#6cf}}
.note{{color:#aaa;font-size:13px;line-height:1.55}}
.warn{{color:#ffcc66}}
</style>
</head>
<body>
<h1>🏝️ 남씨표류기 — Weekly</h1>
<p class="note">
생성: {now}<br>
조건: 가장 최근 주봉 신호가 💎 전환매수 / 미완성 금주 주봉 제외<br>
🇺🇸 미국주식 데이터: {us_status}<br>
₿ 크립토 데이터: {cr_status}
</p>
{section("🇺🇸 미국주식", us)}
{section("₿ 크립토", cr)}
</body>
</html>"""

    (REPORT_DIR / "latest.html").write_text(html, encoding="utf-8")
    return html


def send_telegram(us, cr, us_meta, cr_meta):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[Telegram] secrets not set; skipped.")
        return

    def fmt(title, df):
        lines = [title]

        if df.empty:
            lines.append("조건 충족 없음")
        else:
            for _, r in df.sort_values(
                ["status", "weeks_ago"]
            ).head(25).iterrows():
                lines.append(
                    f"{r['status']} {r['symbol']} | "
                    f"{r['signal_date']} | {r['weeks_ago']}주 | "
                    f"RSI {r['current_rsi']}"
                )

            if len(df) > 25:
                lines.append(f"...외 {len(df) - 25}개")

        return "\n".join(lines)

    meta = (
        f"US {us_meta['coverage']:.0%} "
        f"({us_meta['failed']} 실패) / "
        f"Crypto {cr_meta['source']} {cr_meta['coverage']:.0%} "
        f"({cr_meta['failed']} 실패)"
    )

    msg = (
        "🏝️ 남씨표류기 Weekly\n"
        f"{meta}\n\n"
        + fmt("🇺🇸 미국주식", us)
        + "\n\n"
        + fmt("₿ 크립토", cr)
    )

    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": msg,
            "disable_web_page_preview": True,
        },
        timeout=20,
    ).raise_for_status()


def main():
    prev_us = load_prev("us")
    prev_cr = load_prev("crypto")

    # 한 시장이 실패해도 다른 시장 결과는 반드시 남긴다.
    try:
        us, us_meta = scan_us()
    except Exception as e:
        print(f"[US FATAL] {type(e).__name__}: {e}")
        us = empty_results().drop(columns=["status"])
        us_meta = {
            "source": "Yahoo Finance",
            "total": 0,
            "scanned": 0,
            "failed": 0,
            "coverage": 0.0,
            "ok": False,
        }

    try:
        cr, cr_meta = scan_crypto(300)
    except Exception as e:
        print(f"[CRYPTO FATAL] {type(e).__name__}: {e}")
        cr = empty_results().drop(columns=["status"])
        cr_meta = {
            "source": "none",
            "total": 0,
            "scanned": 0,
            "failed": 0,
            "coverage": 0.0,
            "ok": False,
        }

    us = classify_new(us, "symbol", prev_us)
    cr = classify_new(cr, "symbol", prev_cr)

    if not us.empty:
        us = us.sort_values(
            ["status", "weeks_ago", "signal_date"],
            ascending=[True, True, False],
        )

    if not cr.empty:
        cr = cr.sort_values(
            ["status", "weeks_ago", "signal_date"],
            ascending=[True, True, False],
        )

    us.to_csv(
        REPORT_DIR / "us_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cr.to_csv(
        REPORT_DIR / "crypto_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_html(us, cr, us_meta, cr_meta)

    # 데이터 수집이 너무 많이 실패한 경우 기존 상태를 보존.
    # 그래야 다음 주에 전부 '신규'로 잘못 표시되지 않는다.
    if us_meta.get("ok"):
        save_state("us", set(us["symbol"].tolist()) if not us.empty else set())
    else:
        print("[STATE] US state preserved because coverage was too low.")

    if cr_meta.get("ok"):
        save_state(
            "crypto",
            set(cr["symbol"].tolist()) if not cr.empty else set(),
        )
    else:
        print("[STATE] Crypto state preserved because coverage was too low.")

    try:
        send_telegram(us, cr, us_meta, cr_meta)
    except Exception as e:
        print(f"[Telegram error] {type(e).__name__}: {e}")

    print(
        f"DONE | US eligible={len(us)} coverage={us_meta['coverage']:.1%} | "
        f"Crypto eligible={len(cr)} source={cr_meta['source']} "
        f"coverage={cr_meta['coverage']:.1%}"
    )


if __name__ == "__main__":
    main()
