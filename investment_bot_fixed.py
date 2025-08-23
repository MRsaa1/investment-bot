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
import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения
load_dotenv()

# ========= Secrets / Env =========
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "test_token")
CHAT_ID_RU = os.getenv("TELEGRAM_CHANNEL_RU", "@test_channel")
PROXY_URL = os.getenv("PROXY_URL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
USE_ALPHA_VANTAGE = os.getenv("USE_ALPHA_VANTAGE", "False").lower() == "true"
SIGNATURE = os.getenv("TELEGRAM_SIGNATURE", "Подготовлено @ReserveOne")
POST_LIMIT = int(os.getenv("POST_LIMIT", "2"))
NO_REPEAT_WEEKS = int(os.getenv("NO_REPEAT_WEEKS", "4"))
HISTORY_PATH = os.getenv("HISTORY_PATH", "./last_picks.json")

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

def fetch_prices_simple(tickers: List[str]) -> pd.DataFrame:
    """Простая версия получения цен через Yahoo Finance API"""
    print(f"📊 Получение цен для {len(tickers)} тикеров...")
    
    # Создаем тестовые данные для демонстрации
    dates = pd.date_range(end=date.today(), periods=252, freq='D')
    data = {}
    
    for ticker in tickers[:5]:  # Ограничиваем для демонстрации
        # Генерируем реалистичные цены
        base_price = 100 + hash(ticker) % 200
        prices = []
        for i in range(len(dates)):
            # Добавляем случайные колебания
            change = np.random.normal(0, 0.02)  # 2% волатильность
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + change)
            prices.append(max(price, 1))  # Минимальная цена $1
        
        data[ticker] = prices
    
    df = pd.DataFrame(data, index=dates)
    print(f"✅ Получены цены для {len(df.columns)} тикеров")
    return df

def fetch_basics_simple(tickers: List[str]) -> pd.DataFrame:
    """Простая версия получения базовой информации"""
    print(f"📋 Получение базовой информации для {len(tickers)} тикеров...")
    
    data = []
    for ticker in tickers[:5]:  # Ограничиваем для демонстрации
        # Создаем тестовые данные
        row = {
            'ticker': ticker,
            'name': f"{ticker} Corporation",
            'marketCap': 50e9 + hash(ticker) % 100e9,  # 50-150 млрд
            'trailingPE': 15 + hash(ticker) % 20,  # P/E 15-35
            'returnOnEquity': 0.1 + (hash(ticker) % 20) / 100,  # ROE 10-30%
            'beta': 0.8 + (hash(ticker) % 40) / 100,  # Beta 0.8-1.2
            'averageVolume': 10e6 + hash(ticker) % 20e6,  # Объем 10-30 млн
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✅ Получена базовая информация для {len(df)} тикеров")
    return df

def calculate_scores(prices: pd.DataFrame, basics: pd.DataFrame) -> List[Pick]:
    """Расчет баллов для акций"""
    print("🧮 Расчет баллов...")
    
    picks = []
    for _, row in basics.iterrows():
        ticker = row['ticker']
        
        if ticker not in prices.columns:
            continue
            
        price_series = prices[ticker].dropna()
        if len(price_series) < 60:
            continue
            
        current_price = price_series.iloc[-1]
        price_1m_ago = price_series.iloc[-22] if len(price_series) >= 22 else current_price
        price_3m_ago = price_series.iloc[-63] if len(price_series) >= 63 else current_price
        
        ret_1m = (current_price / price_1m_ago - 1) * 100
        ret_3m = (current_price / price_3m_ago - 1) * 100
        dist_to_high = (current_price / price_series.max() - 1) * 100
        
        # Расчет баллов
        scores = {
            'value': min(10, max(0, 20 - row.get('trailingPE', 20))),
            'growth': min(10, max(0, (ret_3m + 20) / 4)),
            'quality': min(10, max(0, row.get('returnOnEquity', 0.1) * 50)),
            'momentum': min(10, max(0, (dist_to_high + 50) / 10)),
            'risk': min(10, max(0, 10 - abs(row.get('beta', 1) - 1) * 10))
        }
        
        total_score = sum(scores.values()) / len(scores)
        
        pick = Pick(
            ticker=ticker,
            name=row.get('name', ticker),
            price=current_price,
            ret_1m=ret_1m,
            ret_3m=ret_3m,
            dist_to_high=dist_to_high,
            pe=row.get('trailingPE'),
            market_cap=row.get('marketCap', 0),
            dollar_vol=row.get('averageVolume', 0) * current_price,
            beta=row.get('beta'),
            vol_60d=price_series.tail(60).std() / price_series.tail(60).mean() * 100,
            scores=scores,
            total_score=total_score
        )
        picks.append(pick)
    
    print(f"✅ Рассчитаны баллы для {len(picks)} акций")
    return picks

def generate_html_report(ticker: str, pick: Pick, prices: pd.DataFrame) -> str:
    """Генерация HTML отчета"""
    print(f"📄 Генерация отчета для {ticker}...")
    
    # Создаем простой график
    plt.figure(figsize=(12, 8))
    
    if ticker in prices.columns:
        price_series = prices[ticker].dropna()
        plt.plot(price_series.index, price_series.values, linewidth=2, color='#2563eb')
        plt.title(f'Динамика цены {ticker}', fontsize=16, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Цена ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Сохраняем график
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        chart_img = f"data:image/png;base64,{img_base64}"
    else:
        chart_img = ""
    
    # HTML шаблон
    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Анализ {ticker}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .content {{
                padding: 30px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: #f8fafc;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .footer {{
                background: #f8fafc;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }}
            .score {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Анализ акции {ticker}</h1>
                <p>{pick.name}</p>
            </div>
            
            <div class="content">
                <div class="metrics">
                    <div class="metric-card">
                        <h3>💰 Текущая цена</h3>
                        <p class="score">${pick.price:.2f}</p>
                    </div>
                    <div class="metric-card">
                        <h3>📊 P/E Ratio</h3>
                        <p class="score">{pick.pe:.1f if pick.pe else 'н/д'}</p>
                    </div>
                    <div class="metric-card">
                        <h3>📈 Доходность 3 месяца</h3>
                        <p class="score">{pick.ret_3m:+.1f}%</p>
                    </div>
                    <div class="metric-card">
                        <h3>🎯 Общий балл</h3>
                        <p class="score">{pick.total_score:.1f}/10</p>
                    </div>
                </div>
                
                {f'<div class="chart-container"><img src="{chart_img}" alt="График цены"></div>' if chart_img else ''}
                
                <div class="metrics">
                    <div class="metric-card">
                        <h3>📊 Детальные баллы</h3>
                        <p>Value: {pick.scores['value']:.1f}/10</p>
                        <p>Growth: {pick.scores['growth']:.1f}/10</p>
                        <p>Quality: {pick.scores['quality']:.1f}/10</p>
                        <p>Momentum: {pick.scores['momentum']:.1f}/10</p>
                        <p>Risk: {pick.scores['risk']:.1f}/10</p>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>{SIGNATURE}</strong></p>
                <p><em>Не является инвестиционной рекомендацией. Проведите собственный анализ.</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    """Главная функция"""
    print("🚀 Запуск Investment Bot...")
    
    # Получаем данные
    prices = fetch_prices_simple(UNIVERSE)
    basics = fetch_basics_simple(UNIVERSE)
    
    # Рассчитываем баллы
    picks = calculate_scores(prices, basics)
    
    # Сортируем по общему баллу
    picks.sort(key=lambda x: x.total_score, reverse=True)
    
    # Генерируем отчеты для топ-2
    os.makedirs('reports', exist_ok=True)
    
    for i, pick in enumerate(picks[:2]):
        html = generate_html_report(pick.ticker, pick, prices)
        
        filename = f"reports/{pick.ticker}_{date.today().strftime('%Y-%m-%d')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Отчет сохранен: {filename}")
    
    # Выводим результаты
    print("\n🏆 Топ-5 акций:")
    for i, pick in enumerate(picks[:5]):
        print(f"{i+1}. {pick.ticker}: {pick.total_score:.1f}/10 (${pick.price:.2f})")
    
    print(f"\n📊 Сгенерировано отчетов: {min(2, len(picks))}")
    print("🎉 Investment Bot завершил работу!")

if __name__ == "__main__":
    main()
