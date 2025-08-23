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
    print("🚀 Starting Investment Bot...")
    
    # Try the fixed version first
    try:
        from investment_bot_fixed import main as run_bot
        print("🔧 Running fixed investment bot...")
        run_bot()
        return
    except Exception as e:
        print(f"❌ Fixed bot failed: {e}")
    
    # Try original bot
    try:
        from investment_bot import main as run_bot
        print("🔧 Running original bot...")
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
            print("🔧 Running simple test as fallback...")
            run_simple_test()
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        print("🔧 Running simple test as fallback...")
        run_simple_test()

def run_simple_test():
    """Run simple test as fallback"""
    try:
        from simple_test import main as run_test
        run_test()
    except Exception as e:
        print(f"❌ Simple test also failed: {e}")
        print("📋 Available files:")
        os.system("ls -la")
        print("\n📦 Trying to install dependencies manually...")
        install_dependencies()
        print("🔄 Final retry...")
        try:
            from simple_test import main as run_test
            run_test()
        except Exception as e2:
            print(f"❌ Final failure: {e2}")
            print("💡 Please check the error messages above")

if __name__ == "__main__":
    main()
