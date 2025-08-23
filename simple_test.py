#!/usr/bin/env python3
"""
Simple test version without problematic imports
"""

import os
import sys
import json
from datetime import datetime

def main():
    """Simple test function"""
    print("🚀 Simple Investment Bot Test")
    print(f"📅 Current time: {datetime.now()}")
    print("✅ Basic imports working")
    
    # Test basic functionality
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        # Create simple test data
        data = {
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'price': [150.0, 300.0, 2500.0],
            'change': [2.5, -1.2, 5.8]
        }
        df = pd.DataFrame(data)
        print("✅ DataFrame created successfully")
        print(df)
        
    except Exception as e:
        print(f"❌ Pandas error: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        # Create simple plot
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Test Plot')
        plt.savefig('test_plot.png')
        plt.close()
        print("✅ Test plot created successfully")
        
    except Exception as e:
        print(f"❌ Matplotlib error: {e}")
    
    print("\n🎉 Simple test completed!")

if __name__ == "__main__":
    main()
