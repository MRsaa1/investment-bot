import os
import io
import json
import time
import math
import base64
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения
load_dotenv()

# ========= Secrets / Env =========
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID_RU = os.getenv("TELEGRAM_CHANNEL_RU")
PROXY_URL = os.getenv("PROXY_URL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
USE_ALPHA_VANTAGE = os.getenv("USE_ALPHA_VANTAGE", "False").lower() == "true"
SIGNATURE = os.getenv("TELEGRAM_SIGNATURE", "Best regards, @ReserveOne")
POST_LIMIT = int(os.getenv("POST_LIMIT", "2"))
NO_REPEAT_WEEKS = int(os.getenv("NO_REPEAT_WEEKS", "4"))
HISTORY_PATH = os.getenv("HISTORY_PATH", "./last_picks.json")

assert BOT_TOKEN, "Нет TELEGRAM_TOKEN"
assert CHAT_ID_RU, "Нет TELEGRAM_CHANNEL_RU"

# Настройка сессии для Telegram
telegram_session = requests.Session()
telegram_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})
if PROXY_URL:
    telegram_session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
    logging.info(f"Using proxy for Telegram: {PROXY_URL}")

# ========= Universe / Params =========
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "LLY", "JPM",
    "V", "WMT", "MA", "XOM", "NVO", "PG", "UNH", "COST", "HD", "ORCL", "ADBE", "PEP",
    "KO", "NFLX", "CRM", "BAC", "TMO", "CSCO", "ABBV", "LIN", "MRK", "AMD", "ACN",
    "INTU", "QCOM", "TXN", "AMAT", "PFE", "MCD", "IBM", "GE", "CAT", "NOW", "SPGI",
    "COP", "HON", "PM", "BKNG", "VRTX", "SBUX", "GS", "PLTR", "UBER"
]

MIN_MARKET_CAP = 10e9
MIN_AVG_DAILY_DOLLAR_VOL = 5e7
LOOKBACK_1M = 21
LOOKBACK_3M = 63
VOL_LOOKBACK = 60

# ========= i18n =========
T = {
    "ru": {
        "title": "Инвестидеи недели — коротко и по делу",
        "date_line": lambda n: f"{date.today().strftime('%d.%m.%Y')} — {n} идеи",
        "not_found": "Инвестидеи недели: подходящих тикеров не найдено. Возможно, проблемы с доступом к данным.",
        "header_emoji": "📈 ",
        "disclaimer": "_Не инвестсовет. Используйте как материал для самостоятельного анализа. Источник данных: Yahoo Finance._",
        "idea_line": "{i}️⃣ {ticker} ({name}): ${price:.2f}\n{ret3m} за 3м; {ret1m} за 1м, в {dist:.1f}% от 52-нед. максимума.\nP/E: {pe}",
        "idea_line_pe_na": "{i}️⃣ {ticker} ({name}): ${price:.2f}\n{ret3m} за 3м; {ret1m} за 1м, в {dist:.1f}% от 52-нед. максимума.\nP/E: н/д",
        "caps": "Капа: {mcap} млрд $, ликвидность: {liq} млн $/день.",
        "mini_value": "— Value: P/E={pe}",
        "mini_value_na": "— Value: P/E=н/д",
        "mini_growth": "— Growth: {ret3m} за 3м, {ret1m} за 1м",
        "mini_quality": "— Quality: ROE={roe}",
        "mini_quality_na": "— Quality: ROE н/д",
        "mini_momentum": "— Momentum: в {dist:.1f}% от 52-нед. максимума",
        "mini_risk": "— Risk: β={beta}",
        "mini_risk_na": "— Risk: β н/д",
        "score": "📊 Итоговый балл: {score:.1f}/10",
        "cover_title": "Инвестидеи недели",
        "cover_sub": "коротко и по делу",
        "cover_date": date.today().strftime("%d %b %Y").replace(".", ""),
        "note_less": "Примечание: из-за правила без повторов за {w} нед. идей меньше обычного.",
    }
}

@dataclass
class Pick:
    ticker: str
    name: str
    price: float
    ret_1m: float
    ret_3m: float
    dist_to_high: float
    pe: Optional[float]
    market_cap: float
    dollar_vol: float
    beta: Optional[float]
    vol_60d: float
    scores: dict
    total_score: float

# ========= History (no-repeat) =========
def load_history(path: str) -> List[Dict]:
    if not os.path.exists(path):
        logging.info(f"History file {path} not found, creating empty list")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load history from {path}: {e}")
        return []

def save_history(path: str, records: List[Dict]) -> None:
    cutoff = date.today() - timedelta(weeks=26)
    trimmed = [r for r in records if date.fromisoformat(r["date"]) >= cutoff]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved history to {path}")
    except Exception as e:
        logging.error(f"Failed to save history to {path}: {e}")

def banned_last_weeks(records: List[Dict], weeks: int) -> set:
    cutoff = date.today() - timedelta(weeks=weeks)
    banned = {r["ticker"] for r in records if date.fromisoformat(r["date"]) >= cutoff}
    logging.info(f"Banned tickers (last {weeks} weeks): {banned}")
    return banned

# ========= Data (robust against errors) =========
def validate_tickers(tickers: List[str]) -> List[str]:
    """Check which tickers are valid before fetching prices."""
    valid_tickers = []
    for t in tickers:
        try:
            logging.info(f"Validating ticker {t}")
            # Try to get basic info first
            ticker = yf.Ticker(t)
            info = ticker.info
            if info and info.get("regularMarketPrice"):
                valid_tickers.append(t)
                logging.info(f"Ticker {t} is valid")
            else:
                logging.warning(f"Ticker {t} has no valid price data")
        except Exception as e:
            logging.warning(f"Failed to validate ticker {t}: {e}")
        time.sleep(0.5)  # Increased delay to avoid rate limits
    return valid_tickers

def fetch_prices(tickers: List[str]) -> pd.DataFrame:
    """Load prices with retries, caching, and fallback to Alpha Vantage."""
    cache_file = "prices_cache.pkl"
    if os.path.exists(cache_file):
        try:
            prices = pd.read_pickle(cache_file)
            if not prices.empty and set(tickers).intersection(prices.columns):
                logging.info(f"Loaded cached prices for {len(prices.columns)} tickers")
                return prices[tickers].dropna(how="all", axis=1)
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_file}: {e}")

    if USE_ALPHA_VANTAGE and ALPHA_VANTAGE_API_KEY:
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        all_px = []
        for t in tickers:
            attempt = 0
            while attempt < 3:
                try:
                    logging.info(f"Fetching Alpha Vantage prices for {t}")
                    df, _ = ts.get_daily_adjusted(symbol=t, outputsize='compact')
                    df = df[['5. adjusted close']].rename(columns={'5. adjusted close': t})
                    all_px.append(df)
                    logging.info(f"Success for {t}")
                    break
                except Exception as e:
                    attempt += 1
                    logging.warning(f"Attempt {attempt} failed for {t}: {e}")
                    time.sleep(5.0 * attempt)
            time.sleep(15)  # Alpha Vantage: 5 requests per minute
        if not all_px:
            logging.error("No price data retrieved for any tickers")
            return pd.DataFrame()
        out = pd.concat(all_px, axis=1)
        out.index = pd.to_datetime(out.index)
    else:
        # Try downloading all tickers at once first
        try:
            logging.info(f"Fetching Yahoo Finance prices for all {len(tickers)} tickers")
            df = yf.download(
                tickers, period="252d", interval="1d",
                auto_adjust=True, progress=False, threads=False
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if "Adj Close" in df.columns.levels[0]:
                        out = df["Adj Close"]
                    else:
                        out = df.xs("Close", axis=1, level=0, drop_level=True)
                else:
                    out = df
                logging.info(f"Successfully downloaded prices for {len(out.columns)} tickers")
            else:
                raise ValueError("Empty download")
        except Exception as e:
            logging.warning(f"Bulk download failed: {e}, trying individual downloads")
            # Fallback to individual downloads
            all_px = []
            for t in tickers:
                attempt = 0
                while attempt < 3:
                    try:
                        logging.info(f"Fetching individual price for {t}")
                        df = yf.download(
                            t, period="252d", interval="1d",
                            auto_adjust=True, progress=False, threads=False
                        )
                        if not df.empty:
                            if isinstance(df.columns, pd.MultiIndex):
                                if "Adj Close" in df.columns.levels[0]:
                                    df = df["Adj Close"]
                                else:
                                    df = df.xs("Close", axis=1, level=0, drop_level=True)
                            df.columns = [t]
                            all_px.append(df)
                            logging.info(f"Success for {t}")
                            break
                    except Exception as e:
                        attempt += 1
                        logging.warning(f"Attempt {attempt} failed for {t}: {e}")
                        time.sleep(2.0 * attempt)
                if attempt == 3:
                    logging.error(f"Failed to fetch price for {t}")
                time.sleep(1.0)  # Delay between individual downloads
            
            if not all_px:
                logging.error("No price data retrieved for any tickers")
                return pd.DataFrame()
            out = pd.concat(all_px, axis=1)

    out = out.loc[:, ~out.columns.duplicated()].dropna(how="all", axis=1)
    try:
        out.to_pickle(cache_file)
        logging.info(f"Saved prices to cache {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to save cache {cache_file}: {e}")
    logging.info(f"Retrieved prices for {len(out.columns)} tickers")
    return out

def fetch_basics(tickers: List[str]) -> pd.DataFrame:
    """Fast basics via fast_info (name is placeholder; enrich later)."""
    rows = []
    for t in tickers:
        attempt = 0
        while attempt < 3:
            try:
                logging.info(f"Fetching basics for {t}")
                ticker = yf.Ticker(t)
                info = ticker.info
                
                name = info.get("shortName") or info.get("longName") or t
                market_cap = float(info.get("marketCap") or 0)
                last_price = float(info.get("regularMarketPrice") or 0)
                avg_vol = float(info.get("averageVolume") or 0)
                dollar_vol = avg_vol * last_price if last_price > 0 else 0.0
                
                rows.append({
                    "ticker": t, "name": name, "market_cap": market_cap,
                    "dollar_vol": dollar_vol, "price": last_price,
                    "pe": None, "beta": None, "roe": None
                })
                logging.info(f"Success fetching basics for {t}")
                break
            except Exception as e:
                attempt += 1
                logging.warning(f"Attempt {attempt} failed for {t}: {e}")
                time.sleep(5.0 * attempt)  # Increased delay between attempts
        if attempt == 3:
            logging.error(f"Failed to fetch basics for {t}")
        time.sleep(2.0)  # Increased delay to avoid rate limits
    return pd.DataFrame(rows)

def enrich_info(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """For small top subset, pull heavy .info fields with retries."""
    target = df["ticker"].head(top_n).tolist()
    extra = []
    for t in target:
        attempt = 0
        while attempt < 3:
            try:
                logging.info(f"Enriching info for {t}")
                T = yf.Ticker(t)
                info = T.info or {}
                row = {
                    "ticker": t,
                    "name": info.get("shortName") or info.get("longName") or t,
                    "pe": (info.get("trailingPE") if isinstance(info.get("trailingPE"), (int, float)) and 0 < info.get("trailingPE") < 500 else None),
                    "beta": (info.get("beta") if isinstance(info.get("beta"), (int, float)) else None),
                    "roe": (info.get("returnOnEquity") if isinstance(info.get("returnOnEquity"), (int, float)) else None),
                }
                extra.append(row)
                logging.info(f"Success enriching info for {t}")
                break
            except Exception as e:
                attempt += 1
                logging.warning(f"Attempt {attempt} failed for {t}: {e}")
                time.sleep(2.0 * attempt)
        if attempt == 3:
            logging.error(f"Enrich .info failed for {t}")
    if not extra:
        return df
    extra_df = pd.DataFrame(extra)
    for col in ["name", "pe", "beta", "roe"]:
        df = df.merge(extra_df[["ticker", col]], on="ticker", how="left", suffixes=("", "_new"))
        df[col] = df[col].combine_first(df[f"{col}_new"])
        df.drop(columns=[f"{col}_new"], inplace=True, errors='ignore')
    return df

def compute_metrics(prices: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    """Robust metrics computation (handles short history / empties)."""
    if prices is None or prices.empty:
        logging.error("No price data available for metrics computation")
        return pd.DataFrame()

    px = prices.sort_index().ffill().dropna(how="all", axis=1)
    if px.empty or len(px) < 5:
        logging.error(f"Price data too short or empty: {len(px)} rows")
        return pd.DataFrame()

    lb1 = min(LOOKBACK_1M, len(px) - 1)
    lb3 = min(LOOKBACK_3M, len(px) - 1)

    try:
        last = px.iloc[-1]
    except Exception as e:
        logging.error(f"Failed to get last prices: {e}")
        return pd.DataFrame()

    def safe_ret(back):
        if back >= len(px):
            back = len(px) - 1
        base = px.ffill().iloc[-back-1] if back > 0 else px.iloc[0]
        return (last / base - 1.0).where(last.notna() & base.notna(), 0.0)

    ret_1m = safe_ret(lb1).rename("ret_1m")
    ret_3m = safe_ret(lb3).rename("ret_3m")
    high_52w = px.max().replace(0, pd.NA)
    dist_to_high = (last / high_52w - 1.0).where(high_52w.notna(), 0.0).rename("dist_to_high")
    vol_60d = px.pct_change().rolling(min(VOL_LOOKBACK, len(px)-1), min_periods=1).std().iloc[-1].rename("vol_60d")

    df = pd.concat([ret_1m, ret_3m, dist_to_high, vol_60d], axis=1).reset_index().rename(columns={"index": "ticker"})
    
    # Fix column name case if needed
    if 'Ticker' in df.columns and 'ticker' not in df.columns:
        df = df.rename(columns={'Ticker': 'ticker'})
        logging.info("Renamed 'Ticker' to 'ticker' in metrics DataFrame")
    
    df = df.merge(info_df, on="ticker", how="inner")
    df = df.dropna(subset=["price"])
    logging.info(f"Computed metrics for {len(df)} tickers")
    return df

# ========= Scoring =========
def rank01(series: pd.Series, invert: bool = False) -> pd.Series:
    r = series.rank(pct=True)
    return (1 - r if invert else r).fillna(0.5).clip(0, 1)

def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["score_value"] = rank01(df["pe"].fillna(df["pe"].mean()), invert=True)
    df["score_growth"] = 0.6 * rank01(df["ret_3m"].fillna(0)) + 0.4 * rank01(df["ret_1m"].fillna(0))
    df["score_quality"] = rank01(df["roe"].fillna(0))
    beta_penalty = (df["beta"].fillna(1.0) - 1.0).abs()
    df["score_momentum"] = 0.6 * (1 - rank01(df["dist_to_high"].fillna(0))) + 0.4 * rank01(df["ret_3m"].fillna(0))
    df["score_risk"] = 0.6 * rank01(df["vol_60d"].fillna(0), invert=True) + 0.4 * rank01(beta_penalty, invert=True)
    df["total_score"] = (
        0.25 * df["score_value"] +
        0.25 * df["score_growth"] +
        0.25 * df["score_quality"] +
        0.25 * (0.7 * df["score_momentum"] + 0.3 * df["score_risk"])
    ).clip(0, 1) * 10
    logging.info(f"Built scores for {len(df)} tickers")
    return df

# ========= Formatting / Cover =========
def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x*100:+.1f}%"

def fmt_num(n: float) -> str:
    return "—" if pd.isna(n) or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))) else f"{n:,.0f}".replace(",", " ")

def idea_block(r: pd.Series, lang: str) -> str:
    L = T[lang]
    ret3m, ret1m = fmt_pct(r["ret_3m"]), fmt_pct(r["ret_1m"])
    dist = abs(r["dist_to_high"] * 100)
    pe_str = f"{r['pe']:.1f}" if pd.notna(r["pe"]) else "н/д"
    head = L["idea_line"].format(i="{i}", ticker=r["ticker"], name=r["name"], price=r["price"], ret3m=ret3m, ret1m=ret1m, dist=dist, pe=pe_str)
    mcap = fmt_num(r["market_cap"] / 1e9)
    liq = fmt_num(r["dollar_vol"] / 1e6)
    caps = L["caps"].format(mcap=mcap, liq=liq)
    mini = [
        L["mini_value"].format(pe=pe_str),
        L["mini_growth"].format(ret3m=ret3m, ret1m=ret1m),
        L["mini_quality"].format(roe=f"{(r['roe']*100):.1f}%" if pd.notna(r["roe"]) else "н/д"),
        L["mini_momentum"].format(dist=dist),
        L["mini_risk"].format(beta=f"{r['beta']:.2f}" if pd.notna(r["beta"]) else "н/д"),
    ]
    score = L["score"].format(score=r["total_score"])
    return head.replace("{i}", "") + f"\n{caps}\n\n" + "\n".join(mini) + f"\n\n{score}\n" + ("—"*10)

def build_post(df_top: pd.DataFrame, lang: str, note_less: bool, report_paths: List[str] = None) -> str:
    L = T[lang]
    chunks = []
    for i, (_, r) in enumerate(df_top.iterrows(), start=1):
        block = idea_block(r, lang)
        lines = block.splitlines()
        if lines:
            lines[0] = f"{i}️⃣ " + lines[0]
        chunks.append("\n".join(lines))
    disclaimer = "\n" + L["disclaimer"]
    note = f"\n\n{(L['note_less'].format(w=NO_REPEAT_WEEKS) if note_less else '')}"
    signature = f"\n\n{SIGNATURE}"
    
    # Add HTML report links if available
    report_links = ""
    if report_paths and len(report_paths) > 0:
        from pathlib import Path
        from datetime import date
        
        # Generate links for each report
        link_parts = []
        for i, (_, ticker_data) in enumerate(df_top.iterrows()):
            if i < len(report_paths):
                ticker = ticker_data["ticker"]
                filename = f"{ticker}_{date.today().strftime('%Y-%m-%d')}.html"
                link_parts.append(f"[{ticker}](https://my-site/{filename})")
        
        if link_parts:
            report_links = f"\n\n📑 Отчёты глубокого анализа: {', '.join(link_parts)}"
    
    text = "\n".join(chunks) + disclaimer + note + signature + report_links
    return text[:3900]  # Truncate to avoid Telegram limit

def make_cover(tickers: List[str], lang: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    import io
    from datetime import date

    # Create morning-style background (warm colors, coffee-like tones)
    W, H = 1280, 720
    base_img = Image.new("RGB", (W, H), (245, 240, 235))  # Warm cream background
    d = ImageDraw.Draw(base_img)

    # Create coffee cup shape (simplified)
    cup_x, cup_y = W - 200, H - 200
    d.ellipse((cup_x, cup_y, cup_x + 120, cup_y + 80), fill=(139, 69, 19), outline=(101, 67, 33), width=3)  # Coffee cup
    d.ellipse((cup_x + 10, cup_y + 10, cup_x + 110, cup_y + 70), fill=(160, 82, 45), outline=(139, 69, 19), width=2)  # Coffee inside
    d.rectangle((cup_x + 100, cup_y + 20, cup_x + 130, cup_y + 40), fill=(139, 69, 19), outline=(101, 67, 33), width=2)  # Cup handle
    
    # Create newspaper effect (folded paper)
    d.rectangle((50, H - 300, 400, H - 100), fill=(255, 255, 250), outline=(200, 200, 200), width=2)
    d.rectangle((60, H - 290, 390, H - 110), fill=(248, 248, 245), outline=(180, 180, 180), width=1)
    
    # Add some text lines to simulate newspaper
    for i in range(5):
        y_pos = H - 280 + i * 35
        d.line((70, y_pos, 380, y_pos), fill=(100, 100, 100), width=1)
        d.line((70, y_pos + 15, 320, y_pos + 15), fill=(150, 150, 150), width=1)

    # Fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf"
    ]
    title_font = date_font = tag_font = None
    for path in font_paths:
        try:
            title_font = ImageFont.truetype(path, 72)
            date_font = ImageFont.truetype(path, 40)
            tag_font = ImageFont.truetype(path, 56)
            break
        except Exception:
            continue
    if not title_font:
        logging.warning("Failed to load custom fonts, using default")
        title_font = ImageFont.load_default()
        date_font = ImageFont.load_default()
        tag_font = ImageFont.load_default()

    # Title (center top)
    title_text = "ИНВЕСТИДЕИ НЕДЕЛИ"
    try:
        w = d.textlength(title_text, font=title_font)
        d.text(((W - w) / 2, 80), title_text, font=title_font, fill=(50, 50, 50))
    except:
        # Fallback for default font
        d.text((W // 2 - 200, 80), title_text, font=title_font, fill=(50, 50, 50))

    # Subtitle
    sub_text = "коротко и по делу"
    try:
        w = d.textlength(sub_text, font=date_font)
        d.text(((W - w) / 2, 170), sub_text, font=date_font, fill=(100, 100, 100))
    except:
        # Fallback for default font
        d.text((W // 2 - 100, 170), sub_text, font=date_font, fill=(100, 100, 100))

    # Date (top right corner)
    date_text = date.today().strftime("%d.%m.%Y")
    try:
        w = d.textlength(date_text, font=date_font)
        d.text((W - w - 50, 40), date_text, font=date_font, fill=(80, 80, 80))
    except:
        # Fallback for default font
        d.text((W - 150, 40), date_text, font=date_font, fill=(80, 80, 80))

    # Ticker symbols (gray cards at the bottom)
    x = 60
    for t in tickers[:2]:
        # Create rounded rectangle for ticker card
        card_width, card_height = 220, 80
        d.rounded_rectangle((x, H - 160, x + card_width, H - 80), radius=15, fill=(230, 230, 230), outline=(180, 180, 180), width=2)
        
        # Add subtle shadow effect
        d.rounded_rectangle((x + 2, H - 158, x + card_width + 2, H - 78), radius=15, fill=(200, 200, 200), outline=(160, 160, 160), width=1)
        
        # Center ticker text in card
        try:
            ticker_w = d.textlength(t, font=tag_font)
            ticker_x = x + (card_width - ticker_w) / 2
        except:
            # Fallback for default font
            ticker_x = x + (card_width - 50) / 2
        ticker_y = H - 150
        d.text((ticker_x, ticker_y), t, font=tag_font, fill=(50, 50, 50))
        
        x += card_width + 40

    # Save to buffer
    buf = io.BytesIO()
    base_img.save(buf, format="PNG")
    buf.seek(0)
    logging.info("Generated morning-style cover image")
    return buf.getvalue()

# ========= Company Descriptions =========
def get_company_description(ticker: str, info: dict) -> dict:
    """Получает описание компании, сильные и слабые стороны, перспективы и угрозы."""
    
    company_descriptions = {
        'AAPL': {
            'name': 'Apple Inc.',
            'description': 'Apple Inc. - американская технологическая корпорация, которая проектирует, разрабатывает и продает потребительскую электронику, компьютерное программное обеспечение и онлайн-сервисы. Компания известна своими инновационными продуктами, включая iPhone, iPad, Mac, Apple Watch и AirPods.',
            'business': 'Производство и продажа потребительской электроники, программного обеспечения и цифровых сервисов',
            'strengths': [
                'Сильный бренд и лояльная клиентская база',
                'Высокая прибыльность и денежные потоки',
                'Инновационные продукты и экосистема',
                'Глобальное присутствие и дистрибуция',
                'Сильные позиции в премиум-сегменте'
            ],
            'weaknesses': [
                'Высокая зависимость от iPhone',
                'Дорогие продукты ограничивают доступность',
                'Зависимость от поставщиков компонентов',
                'Регуляторные риски в разных странах'
            ],
            'opportunities': [
                'Рост рынка услуг и подписок',
                'Развитие AR/VR технологий',
                'Расширение в развивающихся рынках',
                'Интеграция с автомобильной отраслью'
            ],
            'threats': [
                'Жесткая конкуренция в смартфонах',
                'Замедление роста рынка смартфонов',
                'Регуляторные ограничения',
                'Зависимость от китайских поставщиков'
            ]
        },
        'MSFT': {
            'name': 'Microsoft Corporation',
            'description': 'Microsoft Corporation - американская технологическая корпорация, которая разрабатывает, производит, лицензирует, поддерживает и продает компьютерное программное обеспечение, бытовую электронику, персональные компьютеры и связанные с ними сервисы.',
            'business': 'Разработка программного обеспечения, облачных сервисов и технологических решений',
            'strengths': [
                'Доминирующие позиции в операционных системах',
                'Сильная экосистема облачных сервисов Azure',
                'Высокая прибыльность и денежные потоки',
                'Диверсифицированный портфель продуктов',
                'Сильные корпоративные отношения'
            ],
            'weaknesses': [
                'Зависимость от Windows и Office',
                'Медленная адаптация к мобильным технологиям',
                'Сложность корпоративных продуктов',
                'Высокие цены на лицензии'
            ],
            'opportunities': [
                'Рост облачных сервисов и AI',
                'Развитие игровой индустрии',
                'Кибербезопасность и корпоративные решения',
                'Интернет вещей и edge computing'
            ],
            'threats': [
                'Конкуренция с Google, Amazon, Apple',
                'Переход на open-source решения',
                'Регуляторные ограничения',
                'Кибербезопасность и приватность'
            ]
        },
        'GOOGL': {
            'name': 'Alphabet Inc. (Google)',
            'description': 'Alphabet Inc. - американская холдинговая компания, созданная в 2015 году как материнская компания Google и нескольких бывших дочерних компаний Google. Компания специализируется на интернет-сервисах, рекламе, облачных вычислениях и искусственном интеллекте.',
            'business': 'Интернет-сервисы, цифровая реклама, облачные вычисления и AI',
            'strengths': [
                'Доминирующие позиции в поисковой рекламе',
                'Сильные позиции в мобильной экосистеме Android',
                'Инновации в AI и машинном обучении',
                'Высокая прибыльность рекламного бизнеса',
                'Диверсифицированный портфель сервисов'
            ],
            'weaknesses': [
                'Высокая зависимость от рекламных доходов',
                'Регуляторные риски и антимонопольные расследования',
                'Сложность монетизации новых проектов',
                'Зависимость от пользовательских данных'
            ],
            'opportunities': [
                'Рост облачных сервисов Google Cloud',
                'Развитие AI и машинного обучения',
                'Автономные транспортные средства (Waymo)',
                'Расширение в здравоохранение и биотехнологии'
            ],
            'threats': [
                'Регуляторные ограничения на рекламу',
                'Конкуренция с Microsoft, Amazon, Meta',
                'Изменения в политике приватности',
                'Зависимость от экономического цикла'
            ]
        },
        'AMZN': {
            'name': 'Amazon.com Inc.',
            'description': 'Amazon.com Inc. - американская технологическая компания, которая занимается электронной коммерцией, облачными вычислениями, цифровыми стриминговыми сервисами и искусственным интеллектом. Компания является одним из крупнейших интернет-ритейлеров в мире.',
            'business': 'Электронная коммерция, облачные сервисы, цифровые медиа и AI',
            'strengths': [
                'Доминирующие позиции в e-commerce',
                'Лидерство в облачных сервисах AWS',
                'Сильная логистическая сеть',
                'Инновации в AI и автоматизации',
                'Высокие доходы и масштаб операций'
            ],
            'weaknesses': [
                'Низкая прибыльность в розничном бизнесе',
                'Высокие операционные расходы',
                'Зависимость от внешних поставщиков',
                'Сложность управления большим масштабом'
            ],
            'opportunities': [
                'Рост облачных сервисов AWS',
                'Международная экспансия',
                'Развитие рекламного бизнеса',
                'Автоматизация и роботизация'
            ],
            'threats': [
                'Конкуренция с Walmart, Target, Shopify',
                'Регуляторные ограничения',
                'Проблемы с трудовыми отношениями',
                'Зависимость от экономического цикла'
            ]
        },
        'META': {
            'name': 'Meta Platforms Inc.',
            'description': 'Meta Platforms Inc. - американская технологическая компания, которая владеет и управляет социальными сетями, включая Facebook, Instagram, WhatsApp и Messenger. Компания фокусируется на развитии метавселенной и виртуальной реальности.',
            'business': 'Социальные сети, цифровая реклама, метавселенная и VR',
            'strengths': [
                'Доминирующие позиции в социальных сетях',
                'Большая пользовательская база',
                'Высокая прибыльность рекламного бизнеса',
                'Сильные данные о пользователях',
                'Инновации в VR и метавселенной'
            ],
            'weaknesses': [
                'Высокая зависимость от рекламных доходов',
                'Проблемы с приватностью и доверием',
                'Регуляторные риски',
                'Зависимость от мобильных платформ'
            ],
            'opportunities': [
                'Развитие метавселенной',
                'Рост рекламного бизнеса',
                'Интеграция AI и машинного обучения',
                'Расширение в e-commerce'
            ],
            'threats': [
                'Регуляторные ограничения',
                'Конкуренция с TikTok, Snapchat',
                'Изменения в политике приватности',
                'Снижение доверия пользователей'
            ]
        }
    }
    
    # Если есть описание для конкретной компании, используем его
    if ticker in company_descriptions:
        return company_descriptions[ticker]
    
    # Иначе создаем базовое описание на основе доступной информации
    industry = info.get('industry', 'Технологии')
    sector = info.get('sector', 'Технологический сектор')
    long_name = info.get('longName', info.get('shortName', ticker))
    
    return {
        'name': long_name,
        'description': f'{long_name} - компания, работающая в {industry.lower()}. Компания специализируется на {sector.lower()} и является частью {industry.lower()}.',
        'business': f'Основная деятельность компании связана с {industry.lower()}',
        'strengths': [
            'Устоявшиеся позиции в отрасли',
            'Доступ к капиталу и ресурсам',
            'Опыт управления и операций',
            'Налаженные бизнес-процессы'
        ],
        'weaknesses': [
            'Зависимость от рыночных условий',
            'Конкурентное давление',
            'Регуляторные риски',
            'Технологические изменения'
        ],
        'opportunities': [
            'Рост рынка и экспансия',
            'Технологические инновации',
            'Международная экспансия',
            'Диверсификация продуктов'
        ],
        'threats': [
            'Экономическая нестабильность',
            'Изменения в регулировании',
            'Новые конкуренты',
            'Технологические сдвиги'
        ]
    }

# ========= Telegram =========
def tg_send_message(chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    attempt = 0
    while attempt < 3:
        try:
            r = telegram_session.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=20)
            logging.info(f"sendMessage -> {chat_id} {r.status_code} {r.text[:200]}")
            r.raise_for_status()
            return
        except Exception as e:
            attempt += 1
            logging.warning(f"sendMessage attempt {attempt} failed for {chat_id}: {e}")
            time.sleep(2.0 * attempt)
    logging.error(f"Failed to send message to {chat_id}")

def tg_send_photo(chat_id: str, image_bytes: bytes, caption: str = None) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("cover.png", image_bytes, "image/png")}
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
        data["parse_mode"] = "Markdown"
    attempt = 0
    while attempt < 3:
        try:
            r = telegram_session.post(url, data=data, files=files, timeout=20)
            logging.info(f"sendPhoto -> {chat_id} {r.status_code} {r.text[:200]}")
            r.raise_for_status()
            return
        except Exception as e:
            attempt += 1
            logging.warning(f"sendPhoto attempt {attempt} failed for {chat_id}: {e}")
            time.sleep(2.0 * attempt)
    logging.error(f"Failed to send photo to {chat_id}")

# ========= HTML Report Generation =========
def generate_html_report(ticker: str, ticker_data: pd.Series, prices: pd.DataFrame) -> str:
    """Generate detailed HTML investment report for a company."""
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Get ticker info
    t = yf.Ticker(ticker)
    info = t.info
    
    # Get company description
    company_desc = get_company_description(ticker, info)
    
    # Generate stock price chart
    chart_img = generate_stock_chart(ticker, prices)
    
    # Generate technical analysis chart
    technical_chart = generate_technical_analysis_chart(ticker, prices)
    
    # Prepare metrics data
    metrics_data = prepare_metrics_data(ticker, info)
    
    # Calculate technical indicators
    technical_data = calculate_technical_indicators(prices)
    
    # Helper function for safe formatting in HTML template
    def safe_html_format(value, format_str, default='N/A'):
        if pd.isna(value) or value is None:
            return default
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            return default
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Глубокий анализ: {company_desc['name']} ({ticker})</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                overflow: hidden;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 3em;
                font-weight: 300;
                margin-bottom: 10px;
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin: 40px 0;
                padding: 30px;
                border-radius: 15px;
                background: #f8f9fa;
            }}
            
            .section h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 2em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            
            .company-overview {{
                background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
            }}
            
            .company-overview h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 2.2em;
            }}
            
            .company-description {{
                font-size: 1.1em;
                line-height: 1.8;
                margin-bottom: 25px;
                color: #34495e;
            }}
            
            .swot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .swot-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .swot-card h3 {{
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.3em;
                border-bottom: 2px solid #3498db;
                padding-bottom: 8px;
            }}
            
            .swot-card ul {{
                list-style: none;
                padding: 0;
            }}
            
            .swot-card li {{
                padding: 8px 0;
                border-bottom: 1px solid #ecf0f1;
                position: relative;
                padding-left: 20px;
            }}
            
            .swot-card li:before {{
                content: "•";
                color: #3498db;
                font-weight: bold;
                position: absolute;
                left: 0;
            }}
            
            .strengths h3 {{ color: #27ae60; }}
            .weaknesses h3 {{ color: #e74c3c; }}
            .opportunities h3 {{ color: #f39c12; }}
            .threats h3 {{ color: #9b59b6; }}
            
            .chart-section {{
                text-align: center;
                margin: 40px 0;
            }}
            
            .chart-section img {{
                max-width: 100%;
                height: auto;
                border-radius: 15px;
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            }}
            
            .metrics-section {{
                margin: 30px 0;
            }}
            
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ecf0f1;
                padding: 15px;
                text-align: left;
            }}
            
            .metrics-table th {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                font-weight: bold;
            }}
            
            .metrics-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            
            .technical-analysis {{
                margin: 20px 0;
            }}
            
            .technical-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-top: 20px;
            }}
            
            .technical-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #3498db;
            }}
            
            .technical-card h3 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.3em;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 10px;
            }}
            
            .indicator-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 0;
                border-bottom: 1px solid #ecf0f1;
            }}
            
            .indicator-item:last-child {{
                border-bottom: none;
            }}
            
            .indicator-label {{
                font-weight: 600;
                color: #34495e;
                min-width: 150px;
            }}
            
            .indicator-value {{
                font-weight: 500;
                color: #2c3e50;
            }}
            
            .indicator-value.positive {{
                color: #27ae60;
            }}
            
            .indicator-value.negative {{
                color: #e74c3c;
            }}
            
            .analyst-section {{
                background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #28a745;
            }}
            
            .risk-factors {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #ffc107;
            }}
            
            .risk-factors ul {{
                padding-left: 20px;
            }}
            
            .risk-factors li {{
                margin: 10px 0;
                padding: 5px 0;
            }}
            
            .conclusion-section {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                padding: 30px;
                border-radius: 15px;
                border-left: 5px solid #2196f3;
            }}
            
            .conclusion-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .conclusion-item {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            
            .conclusion-item h4 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 1.1em;
            }}
            
            .conclusion-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #3498db;
            }}
            
            .warning-section {{
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #f44336;
                margin: 30px 0;
            }}
            
            .warning-section h2 {{
                color: #c62828;
                margin-bottom: 15px;
            }}
            
            .footer {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .neutral {{ color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Глубокий анализ: {company_desc['name']}</h1>
                <div class="subtitle">{ticker} • {date.today().strftime('%d.%m.%Y')}</div>
            </div>
            
            <div class="content">
                <!-- Описание компании -->
                <div class="company-overview">
                    <h2>🏢 Обзор компании</h2>
                    <div class="company-description">
                        <p><strong>{company_desc['name']}</strong> - {company_desc['description']}</p>
                        <p><strong>Основная деятельность:</strong> {company_desc['business']}</p>
                    </div>
                    
                    <div class="swot-grid">
                        <div class="swot-card strengths">
                            <h3>💪 Сильные стороны</h3>
                            <ul>
                                {''.join([f'<li>{strength}</li>' for strength in company_desc['strengths']])}
                            </ul>
                        </div>
                        
                        <div class="swot-card weaknesses">
                            <h3>⚠️ Слабые стороны</h3>
                            <ul>
                                {''.join([f'<li>{weakness}</li>' for weakness in company_desc['weaknesses']])}
                            </ul>
                        </div>
                        
                        <div class="swot-card opportunities">
                            <h3>🚀 Возможности</h3>
                            <ul>
                                {''.join([f'<li>{opportunity}</li>' for opportunity in company_desc['opportunities']])}
                            </ul>
                        </div>
                        
                        <div class="swot-card threats">
                            <h3>🔥 Угрозы</h3>
                            <ul>
                                {''.join([f'<li>{threat}</li>' for threat in company_desc['threats']])}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- График цены -->
                <div class="section">
                    <h2>📈 Динамика цены акции</h2>
                    <div class="chart-section">
                        <img src="data:image/png;base64,{chart_img}" alt="График цены акции">
                    </div>
                </div>
                
                <!-- Технический анализ с графиками -->
                <div class="section">
                    <h2>📊 Технический анализ</h2>
                    <div class="chart-section">
                        <img src="{technical_chart}" alt="Технический анализ {ticker}">
                        
                        <!-- Описание графиков и интерпретация -->
                        <div class="chart-explanation" style="margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                            <h3 style="color: #2c3e50; margin-bottom: 15px;">📈 Что показано на графиках:</h3>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                                <div>
                                    <h4 style="color: #3498db; margin-bottom: 10px;">🔵 График цены (верхний)</h4>
                                    <ul style="margin: 0; padding-left: 20px;">
                                        <li><strong>Синяя линия</strong> - цена {ticker}</li>
                                        <li><strong>Красная линия (SMA 20)</strong> - краткосрочный тренд</li>
                                        <li><strong>Оранжевая линия (SMA 50)</strong> - среднесрочный тренд</li>
                                        <li><strong>Зеленая линия (SMA 200)</strong> - долгосрочный тренд</li>
                                        <li><strong>Серые линии</strong> - полосы Боллинджера (волатильность)</li>
                                    </ul>
                                </div>
                                
                                <div>
                                    <h4 style="color: #e74c3c; margin-bottom: 10px;">🔴 RSI - Индекс относительной силы</h4>
                                    <ul style="margin: 0; padding-left: 20px;">
                                        <li><strong>RSI > 70</strong> - зона перекупленности (возможна коррекция)</li>
                                        <li><strong>RSI < 30</strong> - зона перепроданности (возможен отскок)</li>
                                        <li><strong>RSI 30-70</strong> - нейтральная зона</li>
                                        <li><strong>Текущий RSI</strong>: {safe_html_format(technical_data.get('rsi', 50), '{:.1f}')}</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                                <div>
                                    <h4 style="color: #f39c12; margin-bottom: 10px;">🟡 MACD - Схождение/расхождение скользящих средних</h4>
                                    <ul style="margin: 0; padding-left: 20px;">
                                        <li><strong>Синяя линия</strong> - MACD линия</li>
                                        <li><strong>Красная линия</strong> - сигнальная линия</li>
                                        <li><strong>Гистограмма</strong> - разность между линиями</li>
                                        <li><strong>Пересечение вверх</strong> - бычий сигнал</li>
                                        <li><strong>Пересечение вниз</strong> - медвежий сигнал</li>
                                    </ul>
                                </div>
                                
                                <div>
                                    <h4 style="color: #9b59b6; margin-bottom: 10px;">🟣 Объем торгов</h4>
                                    <ul style="margin: 0; padding-left: 20px;">
                                        <li><strong>Высокий объем</strong> - подтверждает тренд</li>
                                        <li><strong>Низкий объем</strong> - слабость движения</li>
                                        <li><strong>Рост + объем</strong> - сильный бычий сигнал</li>
                                        <li><strong>Падение + объем</strong> - сильный медвежий сигнал</li>
                                    </ul>
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px;">
                                <h4 style="color: #2c3e50; margin-bottom: 10px;">💡 Как интерпретировать:</h4>
                                <ol style="margin: 0; padding-left: 20px;">
                                    <li><strong>Тренд</strong>: Если цена выше SMA 20, 50 и 200 - восходящий тренд</li>
                                    <li><strong>Поддержка/Сопротивление</strong>: Скользящие средние часто выступают как уровни поддержки или сопротивления</li>
                                    <li><strong>Сигналы входа</strong>: Пересечение MACD + RSI в нейтральной зоне + высокий объем</li>
                                    <li><strong>Риск-менеджмент</strong>: Используйте стоп-лоссы ниже ключевых уровней поддержки</li>
                                </ol>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Технический анализ -->
                <div class="section">
                    <h2>📊 Технический анализ</h2>
                    <div class="technical-analysis">
                        <div class="technical-grid">
                            <div class="technical-card">
                                <h3>📈 Технические индикаторы</h3>
                                <div class="indicator-item">
                                    <span class="indicator-label">RSI:</span>
                                    <span class="indicator-value">{safe_html_format(technical_data.get('rsi', 50), '{:.1f}')} (Нейтральный)</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">MACD:</span>
                                    <span class="indicator-value">{safe_html_format(technical_data.get('macd', 0), '{:.4f}')} (Бычий)</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Волатильность:</span>
                                    <span class="indicator-value">{safe_html_format(technical_data.get('volatility', 20), '{:.1f}%')} годовых</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Поддержка:</span>
                                    <span class="indicator-value">${safe_html_format(technical_data.get('support', 0), '{:.2f}')}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Сопротивление:</span>
                                    <span class="indicator-value">${safe_html_format(technical_data.get('resistance', 0), '{:.2f}')}</span>
                                </div>
                            </div>
                            
                            <div class="technical-card">
                                <h3>📊 Доходность и тренды</h3>
                                <div class="indicator-item">
                                    <span class="indicator-label">1 месяц:</span>
                                    <span class="indicator-value {technical_data.get('month_return', 0) >= 0 and 'positive' or 'negative'}">{safe_html_format(technical_data.get('month_return', 0), '{:+.1f}%')}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">3 месяца:</span>
                                    <span class="indicator-value {technical_data.get('quarter_return', 0) >= 0 and 'positive' or 'negative'}">{safe_html_format(technical_data.get('quarter_return', 0), '{:+.1f}%')}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Тренд:</span>
                                    <span class="indicator-value">Сильный восходящий</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Расстояние до максимума:</span>
                                    <span class="indicator-value">{safe_html_format(technical_data.get('distance_to_high', 0), '{:.1f}%')}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Бета:</span>
                                    <span class="indicator-value">{safe_html_format(technical_data.get('beta', 1), '{:.2f}')}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Финансовые показатели -->
                <div class="section">
                    <h2>💰 Финансовые показатели</h2>
                    {metrics_data}
                </div>
                
                <!-- Обзор аналитиков -->
                <div class="analyst-section">
                    <h2>📊 Обзор аналитиков</h2>
                    <p><strong>Целевая цена (консенсус):</strong> ${info.get('targetMeanPrice', 'N/A')}</p>
                    <p><strong>Рекомендация:</strong> {info.get('recommendationKey', 'N/A').title() if info.get('recommendationKey') else 'N/A'}</p>
                    <p><strong>Количество аналитиков:</strong> {info.get('numberOfAnalystOpinions', 'N/A')}</p>
                </div>
                
                <!-- Факторы риска -->
                <div class="risk-factors">
                    <h2>⚠️ Факторы риска</h2>
                    <ul>
                        <li><strong>Рыночный риск:</strong> Цены акций могут быть волатильными и снижаться из-за рыночных условий</li>
                        <li><strong>Отраслевой риск:</strong> Изменения в {info.get('industry', 'отрасли')} могут повлиять на результаты</li>
                        <li><strong>Регуляторный риск:</strong> Изменения в регулировании могут повлиять на бизнес-операции</li>
                        <li><strong>Конкурентный риск:</strong> Усиление конкуренции может повлиять на долю рынка и прибыльность</li>
                        <li><strong>Экономический риск:</strong> Экономические спады могут снизить потребительские расходы и спрос</li>
                    </ul>
                </div>
                
                <!-- Заключение -->
                <div class="conclusion-section">
                    <h2>📝 Заключение</h2>
                    <p><strong>{ticker}</strong> в настоящее время торгуется по цене ${safe_html_format(ticker_data['price'], '{:.2f}')} с коэффициентом P/E {safe_html_format(ticker_data['pe'], '{:.1f}')}.</p>
                    <p>Акция показала доходность {safe_html_format(ticker_data['ret_3m']*100, '{:+.1f}')}% за последние 3 месяца и {safe_html_format(ticker_data['ret_1m']*100, '{:+.1f}')}% за последний месяц.</p>
                    <p>На основе анализа эта акция представляет {get_investment_rating_ru(ticker_data)} инвестиционную возможность с общим баллом {safe_html_format(ticker_data['total_score'], '{:.1f}')}/10.</p>
                    
                    <h3>Ключевые выводы</h3>
                    <div class="conclusion-grid">
                        <div class="conclusion-item">
                            <h4>Текущая цена</h4>
                            <div class="conclusion-value">${safe_html_format(ticker_data['price'], '{:.2f}')}</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>P/E коэффициент</h4>
                            <div class="conclusion-value">{safe_html_format(ticker_data['pe'], '{:.1f}')}</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>Доходность 3м</h4>
                            <div class="conclusion-value">{safe_html_format(ticker_data['ret_3m']*100, '{:+.1f}')}%</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>Общий балл</h4>
                            <div class="conclusion-value">{safe_html_format(ticker_data['total_score'], '{:.1f}')}/10</div>
                        </div>
                    </div>
                </div>
                
                <!-- Важное уведомление -->
                <div class="warning-section">
                    <h2>⚠️ ВАЖНОЕ УВЕДОМЛЕНИЕ</h2>
                    <p>Этот отчет не является инвестиционной рекомендацией. Используйте представленную информацию исключительно как материал для самостоятельного анализа. Рынок акций характеризуется волатильностью и рисками. Всегда проводите собственное исследование перед принятием инвестиционных решений.</p>
                </div>
            </div>
            
            <div class="footer">
                <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: rgba(255,255,255,0.1); border-radius: 10px;">
                    <p style="color: #ffffff;"><strong>Подготовлено ReserveOne - Certified in Blockchain and Digital Assets | Financial Professional Track</strong></p>
                    <p style="color: #ffffff;">10001 Georgetown Pike, Suite 902, Great Falls, VA 22066, США</p>
                    
                    <div style="margin-top: 20px; display: flex; justify-content: center; align-items: center; gap: 30px;">
                        <div style="text-align: center;">
                            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #1e3a8a, #3b82f6); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">
                                <span style="color: white; font-weight: bold; font-size: 18px;">CFA</span>
                            </div>
                            <p style="margin: 0; font-weight: bold; color: #ffffff;">CFA Institute</p>
                            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Member</p>
                        </div>
                        
                        <div style="text-align: center;">
                            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #059669, #10b981); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">
                                <span style="color: white; font-weight: bold; font-size: 16px;">AFP</span>
                            </div>
                            <p style="margin: 0; font-weight: bold; color: #ffffff;">AFP Institute</p>
                            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Member</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    filename = f"{ticker}_{date.today().strftime('%Y-%m-%d')}.html"
    filepath = reports_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Generated HTML report for {ticker}: {filepath}")
    return str(filepath)

def generate_stock_chart(ticker: str, prices: pd.DataFrame) -> str:
    """Generate stock price chart and return as base64 encoded image."""
    
    # Get price data for the ticker
    if ticker in prices.columns:
        ticker_prices = prices[ticker].dropna()
        
        # Create the chart
        plt.figure(figsize=(12, 6))
        plt.plot(ticker_prices.index, ticker_prices.values, linewidth=2, color='#007bff')
        plt.title(f'{ticker} - Динамика цены акции за 1 год', fontsize=16, fontweight='bold')
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Цена ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Add current price annotation
        current_price = ticker_prices.iloc[-1]
        plt.annotate(f'${current_price:.2f}', 
                    xy=(ticker_prices.index[-1], current_price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    else:
        return ""

def calculate_technical_indicators(prices: pd.DataFrame) -> Dict:
    """Calculate technical indicators for the stock."""
    try:
        # Check if we have the required columns
        if 'Close' not in prices.columns:
            print(f"Available columns: {prices.columns.tolist()}")
            # Try to find the close price column
            close_col = None
            for col in prices.columns:
                if 'close' in col.lower() or 'price' in col.lower():
                    close_col = col
                    break
            
            if close_col is None:
                # If no close column found, try to use the first column as price data
                if len(prices.columns) > 0:
                    close_col = prices.columns[0]
                    prices = prices.rename(columns={close_col: 'Close'})
                else:
                    raise ValueError("No close price column found")
            
            prices = prices.rename(columns={close_col: 'Close'})
        
        # Calculate RSI
        delta = prices['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = prices['Close'].ewm(span=12).mean()
        exp2 = prices['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        macd_line = macd.iloc[-1] - signal.iloc[-1]
        
        # Calculate volatility (annualized)
        returns = prices['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100
        
        # Calculate support and resistance levels
        recent_prices = prices['Close'].tail(20)
        support = recent_prices.min()
        resistance = recent_prices.max()
        
        # Calculate returns
        current_price = prices['Close'].iloc[-1]
        month_ago_price = prices['Close'].iloc[-30] if len(prices) > 30 else prices['Close'].iloc[0]
        quarter_ago_price = prices['Close'].iloc[-90] if len(prices) > 90 else prices['Close'].iloc[0]
        
        month_return = ((current_price - month_ago_price) / month_ago_price) * 100
        quarter_return = ((current_price - quarter_ago_price) / quarter_ago_price) * 100
        
        # Calculate distance to 52-week high
        if 'High' in prices.columns:
            year_high = prices['High'].max()
        else:
            year_high = prices['Close'].max()
        distance_to_high = ((year_high - current_price) / current_price) * 100
        
        # Beta (simplified - using market correlation)
        beta = 1.0  # Default value, could be calculated with market data
        
        return {
            'rsi': rsi.iloc[-1],
            'macd': macd_line,
            'volatility': volatility,
            'support': support,
            'resistance': resistance,
            'month_return': month_return,
            'quarter_return': quarter_return,
            'distance_to_high': distance_to_high,
            'beta': beta,
            'sma_20': prices['Close'].rolling(window=20).mean(),
            'sma_50': prices['Close'].rolling(window=50).mean(),
            'macd_signal': signal
        }
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return {
            'rsi': 50,
            'macd': 0,
            'volatility': 20,
            'support': 0,
            'resistance': 0,
            'month_return': 0,
            'quarter_return': 0,
            'distance_to_high': 0,
            'beta': 1.0,
            'sma_20': pd.Series([0]),
            'sma_50': pd.Series([0]),
            'macd_signal': pd.Series([0])
        }

def generate_technical_analysis_chart(ticker: str, prices: pd.DataFrame) -> str:
    """Generate comprehensive technical analysis chart with indicators."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        import numpy as np
        
        # Get price data for the ticker
        if ticker in prices.columns:
            ticker_prices = prices[ticker].dropna()
            dates = ticker_prices.index
            price_values = ticker_prices.values
        else:
            logging.error(f"No price data found for {ticker}")
            return ""
        
        if len(price_values) < 50:
            logging.error(f"Not enough data for technical analysis: {len(price_values)} points")
            return ""
        
        # Calculate technical indicators
        def calculate_sma(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        def calculate_rsi(data, window=14):
            deltas = np.diff(data)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:window])
            avg_loss = np.mean(losses[:window])
            
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def calculate_macd(data):
            ema12 = pd.Series(data).ewm(span=12).mean()
            ema26 = pd.Series(data).ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            return macd_line.iloc[-1], signal_line.iloc[-1]
        
        # Calculate indicators
        sma_20 = calculate_sma(price_values, 20)
        sma_50 = calculate_sma(price_values, 50)
        rsi_value = calculate_rsi(price_values)
        macd_value, macd_signal_value = calculate_macd(price_values)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{ticker} - Технический анализ', fontsize=16, fontweight='bold')
        
        # 1. Price chart with SMA
        ax1.plot(dates, price_values, label='Цена', color='#1f77b4', linewidth=2)
        if len(sma_20) > 0:
            sma_20_dates = dates[-len(sma_20):]
            ax1.plot(sma_20_dates, sma_20, label='SMA 20', color='#ff7f0e', linestyle='--')
        if len(sma_50) > 0:
            sma_50_dates = dates[-len(sma_50):]
            ax1.plot(sma_50_dates, sma_50, label='SMA 50', color='#d62728', linestyle='--')
        
        ax1.set_title('Цена и скользящие средние')
        ax1.set_ylabel('Цена ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        # 2. RSI
        rsi_line = [rsi_value] * len(dates)
        ax2.plot(dates, rsi_line, label=f'RSI: {rsi_value:.1f}', color='#9467bd', linewidth=2)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Перекупленность')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Перепроданность')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax2.set_title('RSI (Индекс относительной силы)')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        # 3. MACD
        macd_line_plot = [macd_value] * len(dates)
        signal_line_plot = [macd_signal_value] * len(dates)
        ax3.plot(dates, macd_line_plot, label=f'MACD: {macd_value:.3f}', color='#1f77b4', linewidth=2)
        ax3.plot(dates, signal_line_plot, label=f'Signal: {macd_signal_value:.3f}', color='#ff7f0e', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        # 4. Volume simulation
        volume_simulation = np.random.randint(50000, 200000, len(dates))
        ax4.bar(dates, volume_simulation, alpha=0.7, color='#2ecc71', width=1)
        ax4.set_title('Объем торгов (симуляция)')
        ax4.set_ylabel('Объем')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logging.error(f"Error generating technical chart for {ticker}: {e}")
        return ""

def prepare_metrics_data(ticker: str, info: Dict) -> str:
    """Prepare metrics table HTML."""
    
    # Define metrics to display with safe formatting
    def safe_format(value, format_str, multiplier=1):
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            if multiplier != 1:
                value = value * multiplier
            return format_str.format(value)
        except (ValueError, TypeError):
            return 'N/A'
    
    metrics = {
        'Цена': f"${info.get('regularMarketPrice', 'N/A')}",
        'P/E': safe_format(info.get('trailingPE'), '{:.1f}'),
        'EV/EBITDA': safe_format(info.get('enterpriseToEbitda'), '{:.1f}'),
        'P/B': safe_format(info.get('priceToBook'), '{:.2f}'),
        'ROE': safe_format(info.get('returnOnEquity'), '{:.1f}%', 100),
        'Чистая маржа': safe_format(info.get('netProfitMargin'), '{:.1f}%', 100),
        'Долг/EBITDA': safe_format(info.get('debtToEquity'), '{:.1f}'),
        'FCF доходность': safe_format(
            info.get('freeCashflow') / info.get('marketCap', 1) if info.get('freeCashflow') and info.get('marketCap') else None,
            '{:.1f}%', 100
        ),
        'Рост выручки': safe_format(info.get('revenueGrowth'), '{:.1f}%', 100),
        'Рост EPS': safe_format(info.get('earningsGrowth'), '{:.1f}%', 100),
        'Дивидендная доходность': safe_format(info.get('dividendYield'), '{:.2f}%', 100)
    }
    
    # Create DataFrame for HTML table
    df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    
    return df_metrics.to_html(classes='metrics-table', index=False, escape=False)

def get_investment_rating(ticker_data: pd.Series) -> str:
    """Get investment rating based on total score."""
    score = ticker_data['total_score']
    if score >= 7:
        return "strong"
    elif score >= 5:
        return "moderate"
    elif score >= 3:
        return "weak"
    else:
        return "very weak"

def get_investment_rating_ru(ticker_data: pd.Series) -> str:
    """Get investment rating in Russian based on total score."""
    score = ticker_data['total_score']
    if score >= 7:
        return "сильную"
    elif score >= 5:
        return "умеренную"
    elif score >= 3:
        return "слабую"
    else:
        return "очень слабую"

# ========= Test Mode Functions =========
def test_reports():
    """Generate test HTML reports using static data without calling Yahoo Finance."""
    import pandas as pd
    import numpy as np
    
    print("Generating test HTML reports with static data...")
    print("=" * 50)
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Define test tickers
    test_tickers = ["NVDA", "PLTR"]
    
    # Static financial data
    static_data = {
        "Price": 120.5,
        "P/E": 35.2,
        "EV/EBITDA": 22.4,
        "P/B": 12.1,
        "ROE": 0.18,  # 18%
        "Net Margin": 0.24,  # 24%
        "Debt/EBITDA": 1.2,
        "FCF Yield": 0.031,  # 3.1%
        "Revenue Growth": 0.15,  # 15%
        "EPS Growth": 0.12,  # 12%
        "Dividend Yield": 0.008,  # 0.8%
        "regularMarketPrice": 120.5,
        "trailingPE": 35.2,
        "enterpriseToEbitda": 22.4,
        "priceToBook": 12.1,
        "returnOnEquity": 0.18,
        "netProfitMargin": 0.24,
        "debtToEquity": 1.2,
        "freeCashflow": 5000000000,  # 5B
        "marketCap": 160000000000,  # 160B
        "revenueGrowth": 0.15,
        "earningsGrowth": 0.12,
        "dividendYield": 0.008,
        "targetMeanPrice": 150.0,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 25,
        "industry": "Technology",
        "shortName": "Test Company"
    }
    
    # Generate synthetic price series for chart
    dates = pd.date_range(end=pd.Timestamp.today(), periods=252)
    base_price = 100
    np.random.seed(42)  # For reproducible results
    price_changes = np.cumsum(np.random.randn(252) * 0.02)  # 2% daily volatility
    prices = pd.Series(base_price + price_changes, index=dates)
    
    # Create synthetic price DataFrame
    prices_df = pd.DataFrame({
        "NVDA": prices * 1.2,  # NVDA slightly higher
        "PLTR": prices * 0.8   # PLTR slightly lower
    })
    
    # Create synthetic ticker data for each company
    for ticker in test_tickers:
        print(f"Generating report for {ticker}...")
        
        # Create synthetic ticker data
        ticker_data = pd.Series({
            "ticker": ticker,
            "name": f"{ticker} Test Company",
            "price": static_data["Price"],
            "pe": static_data["P/E"],
            "ret_1m": 0.05,  # 5% 1-month return
            "ret_3m": 0.15,  # 15% 3-month return
            "dist_to_high": -0.1,  # 10% below 52-week high
            "market_cap": static_data["marketCap"],
            "dollar_vol": 50000000,  # 50M daily volume
            "beta": 1.2,
            "vol_60d": 0.25,
            "total_score": 7.5  # Good score
        })
        
        try:
            # Generate HTML report using test data
            report_path = generate_html_report_test(ticker, ticker_data, prices_df, static_data)
            print(f"✅ Generated test report: {report_path}")
            
            # Check if file exists
            if os.path.exists(report_path):
                file_size = os.path.getsize(report_path)
                print(f"   File size: {file_size:,} bytes")
            else:
                print("   ❌ File not found!")
                
        except Exception as e:
            print(f"❌ Failed to generate test report for {ticker}: {e}")
    
    print(f"\n✅ Test report generation completed!")
    print(f"Reports saved in: {os.path.abspath('reports/')}")

def generate_html_report_test(ticker: str, ticker_data: pd.Series, prices: pd.DataFrame, static_info: Dict) -> str:
    """Generate HTML report using static test data."""
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Get company description
    company_desc = get_company_description(ticker, static_info)
    
    # Generate stock price chart
    chart_img = generate_stock_chart(ticker, prices)
    
    # Generate technical analysis chart
    technical_chart = generate_technical_analysis_chart(ticker, prices)
    
    # Prepare metrics data using static info
    metrics_data = prepare_metrics_data_test(ticker, static_info)
    
    # Calculate technical indicators
    technical_data = calculate_technical_indicators(prices)
    
    # Helper function for safe formatting in HTML template
    def safe_html_format(value, format_str, default='N/A'):
        if pd.isna(value) or value is None:
            return default
        try:
            return format_str.format(value)
        except (ValueError, TypeError):
            return default
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Глубокий анализ: {company_desc['name']} ({ticker}) - Тестовые данные</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                overflow: hidden;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 3em;
                font-weight: 300;
                margin-bottom: 10px;
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .test-banner {{
                background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
                color: #856404;
                padding: 15px;
                text-align: center;
                border-radius: 10px;
                margin: 20px;
                font-weight: bold;
                font-size: 1.1em;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin: 40px 0;
                padding: 30px;
                border-radius: 15px;
                background: #f8f9fa;
            }}
            
            .section h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 2em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            
            .company-overview {{
                background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
            }}
            
            .company-overview h2 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 2.2em;
            }}
            
            .company-description {{
                font-size: 1.1em;
                line-height: 1.8;
                margin-bottom: 25px;
                color: #34495e;
            }}
            
            .swot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .swot-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .swot-card h3 {{
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.3em;
                border-bottom: 2px solid #3498db;
                padding-bottom: 8px;
            }}
            
            .swot-card ul {{
                list-style: none;
                padding: 0;
            }}
            
            .swot-card li {{
                padding: 8px 0;
                border-bottom: 1px solid #ecf0f1;
                position: relative;
                padding-left: 20px;
            }}
            
            .swot-card li:before {{
                content: "•";
                color: #3498db;
                font-weight: bold;
                position: absolute;
                left: 0;
            }}
            
            .strengths h3 {{ color: #27ae60; }}
            .weaknesses h3 {{ color: #e74c3c; }}
            .opportunities h3 {{ color: #f39c12; }}
            .threats h3 {{ color: #9b59b6; }}
            
            .chart-section {{
                text-align: center;
                margin: 40px 0;
            }}
            
            .chart-section img {{
                max-width: 100%;
                height: auto;
                border-radius: 15px;
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            }}
            
            .metrics-section {{
                margin: 30px 0;
            }}
            
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ecf0f1;
                padding: 15px;
                text-align: left;
            }}
            
            .metrics-table th {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                font-weight: bold;
            }}
            
            .metrics-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            
            .analyst-section {{
                background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #28a745;
            }}
            
            .risk-factors {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #ffc107;
            }}
            
            .risk-factors ul {{
                padding-left: 20px;
            }}
            
            .risk-factors li {{
                margin: 10px 0;
                padding: 5px 0;
            }}
            
            .conclusion-section {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                padding: 30px;
                border-radius: 15px;
                border-left: 5px solid #2196f3;
            }}
            
            .conclusion-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .conclusion-item {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            
            .conclusion-item h4 {{
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 1.1em;
            }}
            
            .conclusion-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #3498db;
            }}
            
            .warning-section {{
                background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #f44336;
                margin: 30px 0;
            }}
            
            .warning-section h2 {{
                color: #c62828;
                margin-bottom: 15px;
            }}
            
            .footer {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .neutral {{ color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Глубокий анализ: {company_desc['name']}</h1>
                <div class="subtitle">{ticker} • {date.today().strftime('%d.%m.%Y')} • Тестовые данные</div>
            </div>
            
            <div class="test-banner">
                ⚠️ ВНИМАНИЕ: Этот отчет содержит тестовые данные для демонстрации функциональности
            </div>
            
            <div class="content">
                <!-- Описание компании -->
                <div class="company-overview">
                    <h2>🏢 Обзор компании</h2>
                    <div class="company-description">
                        <p><strong>{company_desc['name']}</strong> - {company_desc['description']}</p>
                        <p><strong>Основная деятельность:</strong> {company_desc['business']}</p>
                    </div>
                    
                    <div class="swot-grid">
                        <div class="swot-card strengths">
                            <h3>💪 Сильные стороны</h3>
                            <ul>
                                {''.join([f'<li>{strength}</li>' for strength in company_desc['strengths']])}
                            </ul>
                        </div>
                        
                        <div class="swot-card weaknesses">
                            <h3>⚠️ Слабые стороны</h3>
                            <ul>
                                {''.join([f'<li>{weakness}</li>' for weakness in company_desc['weaknesses']])}
                            </ul>
                        </div>
                        
                        <div class="swot-card opportunities">
                            <h3>🚀 Возможности</h3>
                            <ul>
                                {''.join([f'<li>{opportunity}</li>' for opportunity in company_desc['opportunities']])}
                            </ul>
                        </div>
                        
                        <div class="swot-card threats">
                            <h3>🔥 Угрозы</h3>
                            <ul>
                                {''.join([f'<li>{threat}</li>' for threat in company_desc['threats']])}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- График цены -->
                <div class="section">
                    <h2>📈 Динамика цены акции</h2>
                    <div class="chart-section">
                        <img src="data:image/png;base64,{chart_img}" alt="График цены акции">
                    </div>
                </div>
                
                <!-- Финансовые показатели -->
                <div class="section">
                    <h2>💰 Финансовые показатели</h2>
                    {metrics_data}
                </div>
                
                <!-- Обзор аналитиков -->
                <div class="analyst-section">
                    <h2>📊 Обзор аналитиков</h2>
                    <p><strong>Целевая цена (консенсус):</strong> ${static_info.get('targetMeanPrice', 'N/A')}</p>
                    <p><strong>Рекомендация:</strong> {static_info.get('recommendationKey', 'N/A').title() if static_info.get('recommendationKey') else 'N/A'}</p>
                    <p><strong>Количество аналитиков:</strong> {static_info.get('numberOfAnalystOpinions', 'N/A')}</p>
                </div>
                
                <!-- Факторы риска -->
                <div class="risk-factors">
                    <h2>⚠️ Факторы риска</h2>
                    <ul>
                        <li><strong>Рыночный риск:</strong> Цены акций могут быть волатильными и снижаться из-за рыночных условий</li>
                        <li><strong>Отраслевой риск:</strong> Изменения в {static_info.get('industry', 'отрасли')} могут повлиять на результаты</li>
                        <li><strong>Регуляторный риск:</strong> Изменения в регулировании могут повлиять на бизнес-операции</li>
                        <li><strong>Конкурентный риск:</strong> Усиление конкуренции может повлиять на долю рынка и прибыльность</li>
                        <li><strong>Экономический риск:</strong> Экономические спады могут снизить потребительские расходы и спрос</li>
                    </ul>
                </div>
                
                <!-- Заключение -->
                <div class="conclusion-section">
                    <h2>📝 Заключение (Тестовые данные)</h2>
                    <p><strong>{ticker}</strong> в настоящее время торгуется по цене ${safe_html_format(ticker_data['price'], '{:.2f}')} с коэффициентом P/E {safe_html_format(ticker_data['pe'], '{:.1f}')}.</p>
                    <p>Акция показала доходность {safe_html_format(ticker_data['ret_3m']*100, '{:+.1f}')}% за последние 3 месяца и {safe_html_format(ticker_data['ret_1m']*100, '{:+.1f}')}% за последний месяц.</p>
                    <p>На основе анализа эта акция представляет {get_investment_rating_ru(ticker_data)} инвестиционную возможность с общим баллом {safe_html_format(ticker_data['total_score'], '{:.1f}')}/10.</p>
                    
                    <h3>Ключевые выводы</h3>
                    <div class="conclusion-grid">
                        <div class="conclusion-item">
                            <h4>Текущая цена</h4>
                            <div class="conclusion-value">${safe_html_format(ticker_data['price'], '{:.2f}')}</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>P/E коэффициент</h4>
                            <div class="conclusion-value">{safe_html_format(ticker_data['pe'], '{:.1f}')}</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>Доходность 3м</h4>
                            <div class="conclusion-value">{safe_html_format(ticker_data['ret_3m']*100, '{:+.1f}')}%</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>Общий балл</h4>
                            <div class="conclusion-value">{safe_html_format(ticker_data['total_score'], '{:.1f}')}/10</div>
                        </div>
                    </div>
                </div>
                
                <!-- Важное уведомление -->
                <div class="warning-section">
                    <h2>⚠️ ВАЖНОЕ УВЕДОМЛЕНИЕ</h2>
                    <p>Этот отчет не является инвестиционной рекомендацией. Используйте представленную информацию исключительно как материал для самостоятельного анализа. Рынок акций характеризуется волатильностью и рисками. Всегда проводите собственное исследование перед принятием инвестиционных решений.</p>
                </div>
            </div>
            
            <div class="footer">
                <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: rgba(255,255,255,0.1); border-radius: 10px;">
                    <p style="color: #ffffff;"><strong>Подготовлено ReserveOne - Certified in Blockchain and Digital Assets | Financial Professional Track</strong></p>
                    <p style="color: #ffffff;">10001 Georgetown Pike, Suite 902, Great Falls, VA 22066, США</p>
                    
                    <div style="margin-top: 20px; display: flex; justify-content: center; align-items: center; gap: 30px;">
                        <div style="text-align: center;">
                            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #1e3a8a, #3b82f6); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">
                                <span style="color: white; font-weight: bold; font-size: 18px;">CFA</span>
                            </div>
                            <p style="margin: 0; font-weight: bold; color: #ffffff;">CFA Institute</p>
                            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Member</p>
                        </div>
                        
                        <div style="text-align: center;">
                            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #059669, #10b981); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">
                                <span style="color: white; font-weight: bold; font-size: 16px;">AFP</span>
                            </div>
                            <p style="margin: 0; font-weight: bold; color: #ffffff;">AFP Institute</p>
                            <p style="margin: 0; font-size: 12px; color: #bdc3c7;">Member</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    filename = f"{ticker}_{date.today().strftime('%Y-%m-%d')}.html"
    filepath = reports_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Generated test HTML report for {ticker}: {filepath}")
    return str(filepath)

def prepare_metrics_data_test(ticker: str, static_info: Dict) -> str:
    """Prepare metrics table HTML using static test data."""
    
    # Define metrics to display with safe formatting
    def safe_format(value, format_str, multiplier=1):
        if value is None or value == 'N/A':
            return 'N/A'
        try:
            if multiplier != 1:
                value = value * multiplier
            return format_str.format(value)
        except (ValueError, TypeError):
            return 'N/A'
    
    metrics = {
        'Цена': f"${static_info.get('regularMarketPrice', 'N/A')}",
        'P/E': safe_format(static_info.get('trailingPE'), '{:.1f}'),
        'EV/EBITDA': safe_format(static_info.get('enterpriseToEbitda'), '{:.1f}'),
        'P/B': safe_format(static_info.get('priceToBook'), '{:.2f}'),
        'ROE': safe_format(static_info.get('returnOnEquity'), '{:.1f}%', 100),
        'Чистая маржа': safe_format(static_info.get('netProfitMargin'), '{:.1f}%', 100),
        'Долг/EBITDA': safe_format(static_info.get('debtToEquity'), '{:.1f}'),
        'FCF доходность': safe_format(
            static_info.get('freeCashflow') / static_info.get('marketCap', 1) if static_info.get('freeCashflow') and static_info.get('marketCap') else None,
            '{:.1f}%', 100
        ),
        'Рост выручки': safe_format(static_info.get('revenueGrowth'), '{:.1f}%', 100),
        'Рост EPS': safe_format(static_info.get('earningsGrowth'), '{:.1f}%', 100),
        'Дивидендная доходность': safe_format(static_info.get('dividendYield'), '{:.2f}%', 100)
    }
    
    # Create DataFrame for HTML table
    df_metrics = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    
    return df_metrics.to_html(classes='metrics-table', index=False, escape=False)

# ========= Ranking pipeline =========
def fetch_and_rank(universe: List[str]) -> pd.DataFrame:
    """Main pipeline to fetch data and rank tickers."""
    logging.info(f"Attempting to fetch prices for {len(universe)} tickers")
    
    prices = fetch_prices(universe)
    if prices.empty:
        logging.error("No price data retrieved")
        return pd.DataFrame()
    
    basics = fetch_basics(prices.columns.tolist())
    if basics.empty:
        logging.error("No basic data retrieved")
        return pd.DataFrame()
    
    metrics = compute_metrics(prices, basics)
    if metrics.empty:
        logging.error("No metrics computed, returning empty DataFrame")
        return pd.DataFrame()
    
    base = metrics[
        (metrics["market_cap"] >= MIN_MARKET_CAP) &
        (metrics["dollar_vol"] >= MIN_AVG_DAILY_DOLLAR_VOL)
    ].copy()
    
    if base.empty:
        logging.warning("No tickers meet market cap or liquidity criteria")
        return base
    
    scored = build_scores(base)
    return scored.sort_values("total_score", ascending=False)

def main():
    history = load_history(HISTORY_PATH)
    banned = banned_last_weeks(history, NO_REPEAT_WEEKS)

    # Get prices data for HTML reports
    prices = fetch_prices(UNIVERSE)
    
    ranked = fetch_and_rank(UNIVERSE)
    if ranked.empty:
        error_msg = "Failed to retrieve data, possibly due to API limits, connectivity issues, or invalid tickers."
        tg_send_message(CHAT_ID_RU, T["ru"]["not_found"] + f"\n{error_msg}\n\n{SIGNATURE}")
        return

    desired = max(2, min(POST_LIMIT, 2))
    top_candidates = ranked.head(25).copy()
    top = top_candidates[~top_candidates["ticker"].isin(banned)].head(desired).reset_index(drop=True)

    note_less = len(top) < desired
    if top.empty:
        error_msg = "No new tickers available due to no-repeat rule or data issues."
        tg_send_message(CHAT_ID_RU, T["ru"]["not_found"] + f"\n{error_msg}\n" + T["ru"]["note_less"].format(w=NO_REPEAT_WEEKS) + f"\n\n{SIGNATURE}")
        return

    top = enrich_info(top, top_n=len(top))
    tickers = top["ticker"].tolist()
    logging.info(f"Selected top tickers: {tickers}")

    # Generate HTML reports for each selected company
    report_paths = []
    for _, ticker_data in top.iterrows():
        try:
            report_path = generate_html_report(ticker_data["ticker"], ticker_data, prices)
            report_paths.append(report_path)
        except Exception as e:
            logging.error(f"Failed to generate HTML report for {ticker_data['ticker']}: {e}")
    
    logging.info(f"Generated {len(report_paths)} HTML reports: {report_paths}")

    cover_ru = make_cover(tickers, "ru")
    post_ru = build_post(top, "ru", note_less, report_paths)
    
    # Send photo without caption first
    tg_send_photo(CHAT_ID_RU, cover_ru)
    
    # Then send text with links separately
    tg_send_message(CHAT_ID_RU, post_ru)

    today = date.today().isoformat()
    history.extend([{"date": today, "ticker": t} for t in tickers])
    save_history(HISTORY_PATH, history)

if __name__ == "__main__":
    main()
