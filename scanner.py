from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO

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
        "signal_date": last["date"].date().isoformat(),
        "weeks_ago": max(0, int((latest - last["date"]).days // 7)),
        "current_rsi": round(float(rsi.dropna().iloc[-1]), 1),
        "signal_rsi": round(last["rsi"], 1),
        "close": float(close.iloc[-1]),
    }


# --------------------------------------------------
# 미국 상장 "일반 기업 주식"만 가져오기
# ETF / 워런트 / 유닛 / 권리 / 우선주 등 제외
# --------------------------------------------------

SPECIAL_NAME_WORDS = (
    "WARRANT",
    "WARRANTS",
    "RIGHT",
    "RIGHTS",
    "UNIT",
    "UNITS",
    "PREFERRED",
    "PREFERENCE",
)

# 원본 심볼이 .W .U .R .WS .RT 등으로 끝나는 특수증권 제거
SPECIAL_SYMBOL_SUFFIX = re.compile(r"\.(W|U|R|WS|RT|WT)$", re.IGNORECASE)


def is_normal_company_stock(raw_symbol, security_name=""):
    if pd.isna(raw_symbol):
        return False

    s = str(raw_symbol).strip().upper()
    name = str(security_name or "").strip().upper()

    if not s or s == "NAN":
        return False

    if "FILE CREATION TIME" in s:
        return False

    # Nasdaq 특수증권 표기
    if SPECIAL_SYMBOL_SUFFIX.search(s):
        return False

    # Yahoo에서 정상 티커로 쓰기 곤란한 특수문자
    if "$" in s or "^" in s or "+" in s:
        return False

    # 증권명 자체가 워런트/유닛/권리/우선주인 경우
    if any(word in name for word in SPECIAL_NAME_WORDS):
        return False

    return True


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
        name_col = "Security Name" if "Security Name" in df.columns else None

        if "Test Issue" in df.columns:
            df = df[df["Test Issue"].astype(str).str.upper() != "Y"]

        if "ETF" in df.columns:
            df = df[df["ETF"].astype(str).str.upper() != "Y"]

        for _, row in df.iterrows():
            raw = row.get(sym_col)
            name = row.get(name_col, "") if name_col else ""

            if not is_normal_company_stock(raw, name):
                continue

            s = str(raw).strip().upper()

            # BRK.B 같은 정상 클래스주는 Yahoo 형식 BRK-B로 변환
            s = s.replace(".", "-")

            if 1 <= len(s) <= 10:
                symbols.add(s)

    return sorted(symbols)


def drop_current_equity_week(s: pd.Series):
    # 월~목에는 현재 미완성 주봉이 섞일 수 있어 마지막 봉 제외
    now = datetime.now(timezone.utc)
    if now.weekday() < 4 and len(s) > 1:
        return s.iloc[:-1]
    return s


def extract_yahoo_close(data: pd.DataFrame, tickers: list[str]):
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


def yahoo_download_batch(batch: list[str]):
    """
    속도 우선 버전:
    - 100개씩
    - threads=True
    - 실패한 종목만 1회 추가 재시도
    """
    found = {}

    try:
        data = yf.download(
            tickers=batch,
            period="10y",
            interval="1wk",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=40,
        )
        found.update(extract_yahoo_close(data, batch))
    except Exception as e:
        print(f"[US batch error] {e}")

    missing = [t for t in batch if t not in found]

    # 전체 배치를 계속 반복하지 않고 실패 종목만 딱 한 번 재시도
    if missing:
        time.sleep(1.5)
        try:
            data = yf.download(
                tickers=missing,
                period="10y",
                interval="1wk",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=40,
            )
            found.update(extract_yahoo_close(data, missing))
        except Exception as e:
            print(f"[US retry error] {e}")

    missing = [t for t in batch if t not in found]
    return found, missing


def scan_us():
    tickers = nasdaq_symbols()
    results = []
    failed = []

    batch_size = 100
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    print(f"[US] normal-company universe={len(tickers)}")
    print("[US] ETF/Warrant/Unit/Right/Preferred excluded")

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        batch_no = start // batch_size + 1

        print(f"[US] batch {batch_no}/{total_batches}")

        found, missing = yahoo_download_batch(batch)
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

        # 너무 빠른 연속 호출만 살짝 완화
        time.sleep(0.25)

    scanned = len(tickers) - len(set(failed))
    coverage = scanned / len(tickers) if tickers else 0

    print(
        f"[US SUMMARY] total={len(tickers)} "
        f"scanned={scanned} "
        f"failed={len(set(failed))} "
        f"coverage={coverage:.1%} "
        f"eligible={len(results)}"
    )

    if failed:
        print("[US FAILED SAMPLE]", ", ".join(sorted(set(failed))[:30]))

    df = pd.DataFrame(results)

    if df.empty:
        df = empty_results().drop(columns=["status"])

    return df, {
        "total": len(tickers),
        "scanned": scanned,
        "failed": len(set(failed)),
        "coverage": coverage,
        "ok": coverage >= 0.70,
    }


def classify_new(df: pd.DataFrame, prev: set[str]):
    df = df.copy()

    if "status" not in df.columns:
        df["status"] = pd.Series(dtype="object")

    if df.empty:
        return df

    df["status"] = df["symbol"].apply(
        lambda x: "🆕 신규" if x not in prev else "✅ 유지"
    )

    return df


def load_prev():
    p = STATE_DIR / "us.json"

    if not p.exists():
        return set()

    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_state(vals):
    (STATE_DIR / "us.json").write_text(
        json.dumps(sorted(vals), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_html(us: pd.DataFrame, meta: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if us.empty:
        table = "<p>현재 조건 충족 종목 없음</p>"
    else:
        d = us.copy()
        d["chart"] = d["chart"].apply(
            lambda u: f'<a href="{u}" target="_blank">차트</a>'
        )
        table = d.to_html(index=False, escape=False)

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>남씨표류기 Weekly</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#111;color:#eee;margin:20px}}
h1{{font-size:1.5rem}}
table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}
th,td{{border:1px solid #333;padding:8px;white-space:nowrap}}
th{{background:#222}}
a{{color:#6cf}}
.note{{color:#aaa;font-size:13px;line-height:1.55}}
</style>
</head>
<body>
<h1>🏝️ 남씨표류기 — 미국주식 Weekly</h1>
<p class="note">
생성: {now}<br>
대상: 미국 일반 기업 주식<br>
제외: ETF / 워런트 / 유닛 / 권리 / 우선주 등 특수증권<br>
조건: 가장 최근 주봉 신호가 💎 전환매수<br>
미완성 금주 주봉 제외<br>
데이터 커버리지: {meta['scanned']}/{meta['total']} ({meta['coverage']:.1%}) /
실패 {meta['failed']}
</p>
<h2>💎 전환 종목 ({len(us)})</h2>
{table}
</body>
</html>"""

    (REPORT_DIR / "latest.html").write_text(html, encoding="utf-8")


def send_telegram(us: pd.DataFrame, meta: dict):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[Telegram] secrets not set; skipped.")
        return

    lines = [
        "🏝️ 남씨표류기 미국주식 Weekly",
        f"커버리지 {meta['coverage']:.0%} / 실패 {meta['failed']}",
        "",
    ]

    if us.empty:
        lines.append("💎 조건 충족 종목 없음")
    else:
        for _, r in us.sort_values(
            ["status", "weeks_ago"]
        ).head(40).iterrows():
            lines.append(
                f"{r['status']} {r['symbol']} | "
                f"{r['signal_date']} | {r['weeks_ago']}주 | "
                f"RSI {r['current_rsi']}"
            )

        if len(us) > 40:
            lines.append(f"...외 {len(us) - 40}개")

    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": "\n".join(lines),
            "disable_web_page_preview": True,
        },
        timeout=20,
    ).raise_for_status()


def main():
    prev = load_prev()

    try:
        us, meta = scan_us()
    except Exception as e:
        print(f"[US FATAL] {type(e).__name__}: {e}")
        us = empty_results().drop(columns=["status"])
        meta = {
            "total": 0,
            "scanned": 0,
            "failed": 0,
            "coverage": 0.0,
            "ok": False,
        }

    us = classify_new(us, prev)

    if not us.empty:
        us = us.sort_values(
            ["status", "weeks_ago", "signal_date"],
            ascending=[True, True, False],
        )

    us.to_csv(
        REPORT_DIR / "us_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_html(us, meta)

    if meta.get("ok"):
        save_state(set(us["symbol"].tolist()) if not us.empty else set())
    else:
        print("[STATE] coverage too low; previous state preserved")

    try:
        send_telegram(us, meta)
    except Exception as e:
        print(f"[Telegram error] {type(e).__name__}: {e}")

    print(
        f"DONE | US eligible={len(us)} | "
        f"coverage={meta['coverage']:.1%} | failed={meta['failed']}"
    )


if __name__ == "__main__":
    main()
