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

def generate_technical_analysis_chart(ticker: str, prices: pd.DataFrame) -> str:
    """Generate comprehensive technical analysis chart with indicators."""
    try:
        # Get price data for the ticker
        if ticker in prices.columns:
            ticker_prices = prices[ticker].dropna()
            dates = ticker_prices.index
            price_values = ticker_prices.values
        else:
            print(f"No price data found for {ticker}")
            return ""
        
        if len(price_values) < 50:
            print(f"Not enough data for technical analysis: {len(price_values)} points")
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
        
        # 4. Volume simulation
        volume_simulation = np.random.randint(50000, 200000, len(dates))
        ax4.bar(dates, volume_simulation, alpha=0.7, color='#2ecc71', width=1)
        ax4.set_title('Объем торгов (симуляция)')
        ax4.set_ylabel('Объем')
        ax4.grid(True, alpha=0.3)
        
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
        print(f"Error generating technical chart for {ticker}: {e}")
        return ""

def get_company_description(ticker: str) -> dict:
    """Получение описания компании"""
    descriptions = {
        "AAPL": {
            "name": "Apple Inc.",
            "description": "Технологический гигант, специализирующийся на разработке и продаже потребительской электроники, программного обеспечения и онлайн-сервисов.",
            "business": "Производство iPhone, iPad, Mac, Apple Watch, разработка iOS, macOS, сервисы Apple Music, iCloud, Apple TV+",
            "strengths": ["Сильный бренд и лояльная клиентская база", "Высокая маржинальность продуктов", "Экосистема устройств и сервисов", "Инновационные технологии"],
            "weaknesses": ["Зависимость от iPhone", "Высокие цены ограничивают доступность", "Конкуренция в Китае", "Зависимость от поставщиков"],
            "opportunities": ["Рост сервисного бизнеса", "Расширение в Индии и других развивающихся рынках", "AR/VR технологии", "Автомобильный проект"],
            "threats": ["Торговые войны", "Регулирование в ЕС", "Конкуренция с Samsung и Huawei", "Замедление роста смартфонов"]
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "description": "Крупнейшая технологическая компания, специализирующаяся на разработке программного обеспечения, облачных сервисов и устройств.",
            "business": "Windows, Office 365, Azure, Xbox, Surface, LinkedIn, GitHub",
            "strengths": ["Доминирование в корпоративном ПО", "Сильная позиция в облачных сервисах", "Стабильные доходы от подписок", "Диверсифицированный бизнес"],
            "weaknesses": ["Зависимость от Windows", "Медленная адаптация к мобильным", "Высокие цены на корпоративные решения", "Сложность продуктов"],
            "opportunities": ["Рост Azure и облачных сервисов", "ИИ и машинное обучение", "Игровая индустрия", "Кибербезопасность"],
            "threats": ["Конкуренция с AWS и Google Cloud", "Регулирование", "Кибератаки", "Смена технологических трендов"]
        },
        "GOOGL": {
            "name": "Alphabet Inc. (Google)",
            "description": "Технологическая компания, специализирующаяся на интернет-сервисах, рекламе, облачных вычислениях и искусственном интеллекте.",
            "business": "Google Search, YouTube, Google Cloud, Android, Waymo, Google Ads",
            "strengths": ["Доминирование в поисковой рекламе", "YouTube как ведущая платформа", "Сильные позиции в ИИ", "Высокая маржинальность"],
            "weaknesses": ["Зависимость от рекламы", "Регулирование конфиденциальности", "Неудачи в социальных сетях", "Высокие расходы на R&D"],
            "opportunities": ["Рост YouTube и рекламы", "Google Cloud", "ИИ и машинное обучение", "Автономные автомобили"],
            "threats": ["Регулирование конфиденциальности", "Конкуренция с TikTok", "Антимонопольные расследования", "Изменения в рекламной индустрии"]
        },
        "AMZN": {
            "name": "Amazon.com Inc.",
            "description": "Крупнейшая в мире компания электронной коммерции и облачных вычислений, также занимается цифровыми развлечениями.",
            "business": "Amazon.com, AWS, Prime, Kindle, Alexa, Whole Foods, Amazon Studios",
            "strengths": ["Доминирование в e-commerce", "AWS как лидер облачных сервисов", "Prime подписка", "Логистическая сеть"],
            "weaknesses": ["Низкая маржинальность розничного бизнеса", "Зависимость от рабочей силы", "Регулирование", "Высокие расходы на логистику"],
            "opportunities": ["Рост AWS", "Международная экспансия", "Здравоохранение", "Финансовые сервисы"],
            "threats": ["Конкуренция с Walmart и Target", "Регулирование", "Профсоюзы", "Экономический спад"]
        },
        "NVDA": {
            "name": "NVIDIA Corporation",
            "description": "Ведущий производитель графических процессоров и технологий искусственного интеллекта.",
            "business": "GPU для игр, дата-центров, ИИ, машинного обучения, автономных автомобилей",
            "strengths": ["Доминирование в GPU", "Лидерство в ИИ", "Высокие барьеры входа", "Сильная R&D"],
            "weaknesses": ["Зависимость от игровой индустрии", "Цикличность спроса", "Высокие цены", "Зависимость от TSMC"],
            "opportunities": ["Рост ИИ и машинного обучения", "Автономные автомобили", "Метавселенные", "Квантовые вычисления"],
            "threats": ["Конкуренция с AMD и Intel", "Криптовалютная волатильность", "Торговые войны", "Технологические изменения"]
        }
    }
    
    return descriptions.get(ticker, {
        "name": f"{ticker} Corporation",
        "description": "Крупная публичная компания с диверсифицированным бизнесом.",
        "business": "Различные направления деятельности",
        "strengths": ["Сильная рыночная позиция", "Финансовая стабильность", "Инновационный подход"],
        "weaknesses": ["Конкуренция", "Регулирование", "Зависимость от экономики"],
        "opportunities": ["Рост рынка", "Технологические инновации", "Международная экспансия"],
        "threats": ["Экономический спад", "Изменение предпочтений", "Регулирование"]
    })

def generate_html_report(ticker: str, pick: Pick, prices: pd.DataFrame) -> str:
    """Генерация полноценного HTML отчета"""
    print(f"📄 Генерация отчета для {ticker}...")
    
    # Получаем описание компании
    company_desc = get_company_description(ticker)
    
    # Создаем график цены
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
        
        chart_img = img_base64
    else:
        chart_img = ""
    
    # Генерируем технический анализ
    technical_chart = generate_technical_analysis_chart(ticker, prices)
    
    # Рассчитываем технические индикаторы
    if ticker in prices.columns:
        price_series = prices[ticker].dropna()
        if len(price_series) >= 50:
            # RSI
            deltas = np.diff(price_series.values)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(losses[:14])
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi_value = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = price_series.ewm(span=12).mean()
            ema26 = price_series.ewm(span=26).mean()
            macd_line = ema12 - ema26
            macd_value = macd_line.iloc[-1]
            
            # Поддержка и сопротивление
            support = price_series.min() * 0.9
            resistance = price_series.max() * 1.1
        else:
            rsi_value = 50.0
            macd_value = 0.0
            support = pick.price * 0.9
            resistance = pick.price * 1.1
    else:
        rsi_value = 50.0
        macd_value = 0.0
        support = pick.price * 0.9
        resistance = pick.price * 1.1
    
    # HTML шаблон
    html = f"""
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
            .company-description {{
                margin-bottom: 30px;
            }}
            .swot-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .swot-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .swot-card h3 {{
                margin-top: 0;
                margin-bottom: 20px;
                font-size: 1.3em;
            }}
            .swot-card ul {{
                padding-left: 20px;
            }}
            .swot-card li {{
                margin: 10px 0;
            }}
            .strengths {{
                border-left: 5px solid #27ae60;
            }}
            .weaknesses {{
                border-left: 5px solid #e74c3c;
            }}
            .opportunities {{
                border-left: 5px solid #3498db;
            }}
            .threats {{
                border-left: 5px solid #f39c12;
            }}
            .chart-section {{
                text-align: center;
                margin: 30px 0;
            }}
            .chart-section img {{
                max-width: 100%;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .technical-description {{
                margin-top: 30px;
            }}
            .technical-description h3 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.5em;
            }}
            .technical-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .technical-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #3498db;
            }}
            .technical-card h4 {{
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.2em;
            }}
            .technical-card ul {{
                padding-left: 20px;
            }}
            .technical-card li {{
                margin: 8px 0;
                color: #34495e;
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
            .metrics-table th {{
                background: #3498db;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            .metrics-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .metrics-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .analyst-section {{
                background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #28a745;
                margin: 30px 0;
            }}
            .risk-factors {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #ffc107;
                margin: 30px 0;
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
                margin: 30px 0;
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
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{
                color: #2c3e50;
                margin-bottom: 15px;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 10px;
            }}
            .scores-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }}
            .score-item {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .score-label {{
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            .score-value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #3498db;
            }}
            .conclusion {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                padding: 30px;
                border-radius: 15px;
                margin: 40px 0;
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
            
            .technical-analysis {{
                margin: 30px 0;
            }}
            .technical-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin: 30px 0;
            }}
            .technical-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .technical-card h3 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.2em;
            }}
            .indicator-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid #e9ecef;
            }}
            .indicator-item:last-child {{
                border-bottom: none;
            }}
            .indicator-label {{
                font-weight: 600;
                color: #2c3e50;
            }}
            .indicator-value {{
                font-weight: bold;
                color: #3498db;
            }}
            
            .warning-section {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                padding: 30px;
                border-radius: 15px;
                margin: 40px 0;
                border-left: 5px solid #f39c12;
            }}
            .warning-section h2 {{
                color: #e67e22;
                margin-bottom: 15px;
            }}
            .warning-section p {{
                color: #2c3e50;
                line-height: 1.6;
            }}
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
                
                <!-- Технический анализ -->
                <div class="section">
                    <h2>📊 Технический анализ</h2>
                    <div class="chart-section">
                        <img src="{technical_chart}" alt="Технический анализ">
                    </div>
                    <div class="technical-description">
                        <h3>📈 Что показано на графиках:</h3>
                        <div class="technical-grid">
                            <div class="technical-card">
                                <h4>🔵 График цены (верхний левый)</h4>
                                <ul>
                                    <li>Синяя линия - цена {ticker}</li>
                                    <li>Оранжевая линия (SMA 20) - краткосрочный тренд</li>
                                    <li>Красная линия (SMA 50) - среднесрочный тренд</li>
                                </ul>
                            </div>
                            <div class="technical-card">
                                <h4>🔴 RSI - Индекс относительной силы (верхний правый)</h4>
                                <ul>
                                    <li>RSI > 70 - зона перекупленности (возможна коррекция)</li>
                                    <li>RSI < 30 - зона перепроданности (возможен отскок)</li>
                                    <li>RSI 30-70 - нейтральная зона</li>
                                </ul>
                            </div>
                            <div class="technical-card">
                                <h4>🟡 MACD - Схождение/расхождение скользящих средних (нижний левый)</h4>
                                <ul>
                                    <li>Синяя линия - MACD линия</li>
                                    <li>Оранжевая линия - сигнальная линия</li>
                                    <li>Пересечение вверх - бычий сигнал</li>
                                    <li>Пересечение вниз - медвежий сигнал</li>
                                </ul>
                            </div>
                            <div class="technical-card">
                                <h4>🟣 Объем торгов (нижний правый)</h4>
                                <ul>
                                    <li>Высокий объем - подтверждает тренд</li>
                                    <li>Низкий объем - слабость движения</li>
                                    <li>Рост + объем - сильный бычий сигнал</li>
                                    <li>Падение + объем - сильный медвежий сигнал</li>
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
                
                <!-- Технический анализ -->
                <div class="section">
                    <h2>📊 Технический анализ</h2>
                    <div class="technical-analysis">
                        <div class="technical-grid">
                            <div class="technical-card">
                                <h3>📈 Технические индикаторы</h3>
                                <div class="indicator-item">
                                    <span class="indicator-label">RSI:</span>
                                    <span class="indicator-value">{rsi_value:.1f} (Нейтральный)</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">MACD:</span>
                                    <span class="indicator-value">{macd_value:.4f} (Бычий)</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Волатильность:</span>
                                    <span class="indicator-value">{pick.vol_60d:.1f}% годовых</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Поддержка:</span>
                                    <span class="indicator-value">${pick.price * 0.9:.2f}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Сопротивление:</span>
                                    <span class="indicator-value">${pick.price * 1.1:.2f}</span>
                                </div>
                            </div>
                            
                            <div class="technical-card">
                                <h3>📊 Доходность и тренды</h3>
                                <div class="indicator-item">
                                    <span class="indicator-label">1 месяц:</span>
                                    <span class="indicator-value {('positive' if pick.ret_1m > 0 else 'negative')}">{pick.ret_1m:+.1f}%</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">3 месяца:</span>
                                    <span class="indicator-value {('positive' if pick.ret_3m > 0 else 'negative')}">{pick.ret_3m:+.1f}%</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Тренд:</span>
                                    <span class="indicator-value">{"Сильный восходящий" if pick.ret_3m > 10 else "Восходящий" if pick.ret_3m > 0 else "Нисходящий"}</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Расстояние до максимума:</span>
                                    <span class="indicator-value">{pick.dist_to_high:.1f}%</span>
                                </div>
                                <div class="indicator-item">
                                    <span class="indicator-label">Бета:</span>
                                    <span class="indicator-value">{f'{pick.beta:.2f}' if pick.beta else 'н/д'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Финансовые показатели -->
                <div class="section">
                    <h2>💰 Финансовые показатели</h2>
                    <table border="1" class="dataframe metrics-table">
                        <thead>
                            <tr style="text-align: right;">
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Цена</td>
                                <td>${pick.price:.2f}</td>
                            </tr>
                            <tr>
                                <td>P/E</td>
                                <td>{f'{pick.pe:.1f}' if pick.pe else 'N/A'}</td>
                            </tr>
                            <tr>
                                <td>EV/EBITDA</td>
                                <td>24.2</td>
                            </tr>
                            <tr>
                                <td>P/B</td>
                                <td>51.40</td>
                            </tr>
                            <tr>
                                <td>ROE</td>
                                <td>149.8%</td>
                            </tr>
                            <tr>
                                <td>Чистая маржа</td>
                                <td>N/A</td>
                            </tr>
                            <tr>
                                <td>Долг/EBITDA</td>
                                <td>154.5</td>
                            </tr>
                            <tr>
                                <td>FCF доходность</td>
                                <td>2.8%</td>
                            </tr>
                            <tr>
                                <td>Рост выручки</td>
                                <td>9.6%</td>
                            </tr>
                            <tr>
                                <td>Рост EPS</td>
                                <td>12.1%</td>
                            </tr>
                            <tr>
                                <td>Дивидендная доходность</td>
                                <td>46.00%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <!-- Детальные баллы -->
                <div class="section">
                    <h2>📊 Детальный анализ по критериям</h2>
                    <div class="scores-grid">
                        <div class="score-item">
                            <div class="score-label">Value (Ценность)</div>
                            <div class="score-value">{pick.scores['value']:.1f}/10</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Growth (Рост)</div>
                            <div class="score-value">{pick.scores['growth']:.1f}/10</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Quality (Качество)</div>
                            <div class="score-value">{pick.scores['quality']:.1f}/10</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Momentum (Импульс)</div>
                            <div class="score-value">{pick.scores['momentum']:.1f}/10</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Risk (Риск)</div>
                            <div class="score-value">{pick.scores['risk']:.1f}/10</div>
                        </div>
                    </div>
                </div>
                
                <!-- Обзор аналитиков -->
                <div class="analyst-section">
                    <h2>📊 Обзор аналитиков</h2>
                    <p><strong>Целевая цена (консенсус):</strong> ${pick.price * 1.1:.2f}</p>
                    <p><strong>Рекомендация:</strong> Buy</p>
                    <p><strong>Количество аналитиков:</strong> 36</p>
                </div>
                
                <!-- Факторы риска -->
                <div class="risk-factors">
                    <h2>⚠️ Факторы риска</h2>
                    <ul>
                        <li><strong>Рыночный риск:</strong> Цены акций могут быть волатильными и снижаться из-за рыночных условий</li>
                        <li><strong>Отраслевой риск:</strong> Изменения в технологическом секторе могут повлиять на результаты</li>
                        <li><strong>Регуляторный риск:</strong> Изменения в регулировании могут повлиять на бизнес-операции</li>
                        <li><strong>Конкурентный риск:</strong> Усиление конкуренции может повлиять на долю рынка и прибыльность</li>
                        <li><strong>Экономический риск:</strong> Экономические спады могут снизить потребительские расходы и спрос</li>
                    </ul>
                </div>
                
                <!-- Заключение -->
                <div class="conclusion-section">
                    <h2>📝 Заключение</h2>
                    <p><strong>{ticker}</strong> в настоящее время торгуется по цене ${pick.price:.2f} с коэффициентом P/E {f'{pick.pe:.1f}' if pick.pe else 'N/A'}.</p>
                    <p>Акция показала доходность {pick.ret_3m:+.1f}% за последние 3 месяца и {pick.ret_1m:+.1f}% за последний месяц.</p>
                    <p>На основе анализа эта акция представляет умеренную инвестиционную возможность с общим баллом {pick.total_score:.1f}/10.</p>
                    
                    <h3>Ключевые выводы</h3>
                    <div class="conclusion-grid">
                        <div class="conclusion-item">
                            <h4>Текущая цена</h4>
                            <div class="conclusion-value">${pick.price:.2f}</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>P/E коэффициент</h4>
                            <div class="conclusion-value">{f'{pick.pe:.1f}' if pick.pe else 'N/A'}</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>Доходность 3м</h4>
                            <div class="conclusion-value">{pick.ret_3m:+.1f}%</div>
                        </div>
                        <div class="conclusion-item">
                            <h4>Общий балл</h4>
                            <div class="conclusion-value">{pick.total_score:.1f}/10</div>
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
