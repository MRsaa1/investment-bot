#!/usr/bin/env python3
"""
Main entry point for Investment Bot on Replit
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    os.system("pip install --upgrade pip")
    os.system("pip install -r requirements.txt")
    print("✅ Dependencies installed")

def main():
    """Main function to run the investment bot"""
    try:
        # Import and run the main bot
        from investment_bot import main as run_bot
        print("🚀 Starting Investment Bot...")
        run_bot()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        install_dependencies()
        print("🔄 Retrying...")
        try:
            from investment_bot import main as run_bot
            run_bot()
        except Exception as e2:
            print(f"❌ Still failing: {e2}")
            print("🔧 Running in test mode...")
            run_test_mode()
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        print("🔧 Running in test mode...")
        run_test_mode()

def run_test_mode():
    """Run in test mode with HTML report generation"""
    try:
        from test_html_report import main as run_test
        run_test()
    except Exception as e:
        print(f"❌ Test mode also failed: {e}")
        print("📋 Available files:")
        os.system("ls -la")
        print("\n📦 Trying to install dependencies manually...")
        install_dependencies()
        print("🔄 Final retry...")
        try:
            from test_html_report import main as run_test
            run_test()
        except Exception as e2:
            print(f"❌ Final failure: {e2}")
            print("💡 Please check the error messages above")

if __name__ == "__main__":
    main()
