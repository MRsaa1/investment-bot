#!/usr/bin/env python3
"""
Unit tests for investment bot
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Добавляем путь к модулю
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from investment_bot import (
    rank01, build_scores, compute_metrics, 
    fmt_pct, fmt_num, idea_block, banned_last_weeks
)

class TestInvestmentBot(unittest.TestCase):
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.test_prices = pd.DataFrame({
            'AAPL': [100, 105, 110, 108, 112],
            'MSFT': [200, 210, 220, 215, 225]
        }, index=pd.date_range('2025-01-01', periods=5))
        
        self.test_info = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'name': ['Apple Inc.', 'Microsoft Corp.'],
            'price': [112.0, 225.0],
            'market_cap': [2e12, 3e12],
            'dollar_vol': [1e9, 1.5e9],
            'pe': [25.0, 30.0],
            'beta': [1.2, 1.1],
            'roe': [0.15, 0.18]
        })
    
    def test_rank01(self):
        """Тест функции ранжирования"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = rank01(series)
        
        # Проверяем, что результат в диапазоне [0, 1]
        self.assertTrue(all(0 <= x <= 1 for x in result))
        
        # Проверяем, что большее значение имеет больший ранг
        self.assertGreater(result.iloc[4], result.iloc[0])
    
    def test_rank01_invert(self):
        """Тест функции ранжирования с инверсией"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = rank01(series, invert=True)
        
        # Проверяем, что результат в диапазоне [0, 1]
        self.assertTrue(all(0 <= x <= 1 for x in result))
        
        # Проверяем, что меньшее значение имеет больший ранг при инверсии
        self.assertGreater(result.iloc[0], result.iloc[4])
    
    def test_fmt_pct(self):
        """Тест форматирования процентов"""
        self.assertEqual(fmt_pct(0.15), "+15.0%")
        self.assertEqual(fmt_pct(-0.05), "-5.0%")
        self.assertEqual(fmt_pct(np.nan), "—")
    
    def test_fmt_num(self):
        """Тест форматирования чисел"""
        self.assertEqual(fmt_num(1000000), "1 000 000")
        self.assertEqual(fmt_num(1234), "1 234")
        self.assertEqual(fmt_num(np.nan), "—")
    
    def test_compute_metrics(self):
        """Тест вычисления метрик"""
        metrics = compute_metrics(self.test_prices, self.test_info)
        
        # Проверяем, что метрики вычислены
        self.assertFalse(metrics.empty)
        self.assertIn('ret_1m', metrics.columns)
        self.assertIn('ret_3m', metrics.columns)
        self.assertIn('dist_to_high', metrics.columns)
        self.assertIn('vol_60d', metrics.columns)
    
    def test_build_scores(self):
        """Тест построения оценок"""
        # Создаем тестовые данные с метриками
        test_data = self.test_info.copy()
        test_data['ret_1m'] = [0.05, 0.08]
        test_data['ret_3m'] = [0.12, 0.15]
        test_data['dist_to_high'] = [-0.05, -0.03]
        test_data['vol_60d'] = [0.02, 0.025]
        
        scored = build_scores(test_data)
        
        # Проверяем, что оценки построены
        self.assertIn('score_value', scored.columns)
        self.assertIn('score_growth', scored.columns)
        self.assertIn('score_quality', scored.columns)
        self.assertIn('score_momentum', scored.columns)
        self.assertIn('score_risk', scored.columns)
        self.assertIn('total_score', scored.columns)
        
        # Проверяем, что итоговая оценка в диапазоне [0, 10]
        self.assertTrue(all(0 <= score <= 10 for score in scored['total_score']))
    
    def test_banned_last_weeks(self):
        """Тест функции исключения повторений"""
        from datetime import date, timedelta
        
        # Создаем тестовые записи истории
        records = [
            {"date": (date.today() - timedelta(days=7)).isoformat(), "ticker": "AAPL"},
            {"date": (date.today() - timedelta(days=14)).isoformat(), "ticker": "MSFT"},
            {"date": (date.today() - timedelta(days=35)).isoformat(), "ticker": "GOOGL"}
        ]
        
        # Тестируем исключение за последние 2 недели
        banned = banned_last_weeks(records, 2)
        self.assertIn("AAPL", banned)
        self.assertIn("MSFT", banned)
        self.assertNotIn("GOOGL", banned)
    
    def test_idea_block(self):
        """Тест генерации блока идеи"""
        # Создаем тестовые данные
        test_row = pd.Series({
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
            'price': 150.0,
            'ret_1m': 0.05,
            'ret_3m': 0.12,
            'dist_to_high': -0.08,
            'pe': 25.0,
            'market_cap': 2e12,
            'dollar_vol': 1e9,
            'beta': 1.2,
            'vol_60d': 0.02,
            'roe': 0.15,
            'total_score': 7.5
        })
        
        result = idea_block(test_row, "ru")
        
        # Проверяем, что блок содержит необходимую информацию
        self.assertIn("AAPL", result)
        self.assertIn("Apple Inc.", result)
        self.assertIn("$150.00", result)
        self.assertIn("7.5/10", result)
    
    @patch('investment_bot.yf.Ticker')
    def test_fetch_basics_mock(self, mock_ticker):
        """Тест получения базовой информации с моком"""
        # Настраиваем мок
        mock_info = {
            'shortName': 'Apple Inc.',
            'marketCap': 2e12,
            'regularMarketPrice': 150.0,
            'averageVolume': 1e8
        }
        mock_ticker.return_value.info = mock_info
        
        # Здесь можно добавить тест fetch_basics если функция доступна
        pass

if __name__ == '__main__':
    unittest.main()
