#!/usr/bin/env python3
"""
Test script for HTML report generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from investment_bot import generate_html_report, fetch_prices, fetch_and_rank
import pandas as pd

def test_html_report():
    """Test HTML report generation for a few tickers"""
    print("Testing HTML Report Generation...")
    print("=" * 50)
    
    # Test with a small subset
    test_tickers = ["AAPL", "MSFT"]
    
    try:
        # Fetch data
        print("Fetching price data...")
        prices = fetch_prices(test_tickers)
        
        print("Ranking tickers...")
        ranked = fetch_and_rank(test_tickers)
        
        if ranked.empty:
            print("❌ No data retrieved for ranking")
            return
        
        print(f"✅ Successfully ranked {len(ranked)} tickers")
        
        # Generate HTML reports for top 2
        top_2 = ranked.head(2)
        
        for _, ticker_data in top_2.iterrows():
            ticker = ticker_data["ticker"]
            print(f"\nGenerating HTML report for {ticker}...")
            
            try:
                report_path = generate_html_report(ticker, ticker_data, prices)
                print(f"✅ Generated HTML report: {report_path}")
                
                # Check if file exists
                if os.path.exists(report_path):
                    file_size = os.path.getsize(report_path)
                    print(f"   File size: {file_size:,} bytes")
                else:
                    print("   ❌ File not found!")
                    
            except Exception as e:
                print(f"❌ Failed to generate report for {ticker}: {e}")
        
        print(f"\n✅ HTML report generation test completed!")
        print(f"Reports saved in: {os.path.abspath('reports/')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_html_report()
