#!/usr/bin/env python3
"""
Main entry point for Investment Bot on Replit
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main function to run the investment bot"""
    try:
        # Import and run the main bot
        from investment_bot import main as run_bot
        print("🚀 Starting Investment Bot...")
        run_bot()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Installing dependencies...")
        os.system("pip install -r requirements.txt")
        print("🔄 Retrying...")
        from investment_bot import main as run_bot
        run_bot()
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        print("🔧 Running in test mode...")
        from test_html_report import main as run_test
        run_test()

if __name__ == "__main__":
    main()
