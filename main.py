#!/usr/bin/env python3
"""
Main entry point for Investment Bot on Replit
Updated for better error handling and logging
"""

import os
import sys
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        os.system("pip install --upgrade pip")
        os.system("pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")

def main():
    """Main function to run the investment bot"""
    print("🚀 Starting Investment Bot on Replit...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    # Try the fixed version first (our updated version)
    try:
        print("🔧 Attempting to run fixed investment bot...")
        from investment_bot_fixed import main as run_bot
        run_bot()
        print("✅ Fixed bot completed successfully!")
        return
    except ImportError as e:
        print(f"❌ Import error in fixed bot: {e}")
        print("📦 Installing dependencies...")
        install_dependencies()
        print("🔄 Retrying fixed bot...")
        try:
            from investment_bot_fixed import main as run_bot
            run_bot()
            print("✅ Fixed bot completed successfully after retry!")
            return
        except Exception as e2:
            print(f"❌ Fixed bot still failing: {e2}")
            traceback.print_exc()
    except Exception as e:
        print(f"❌ Error running fixed bot: {e}")
        traceback.print_exc()
    
    # Try original bot as fallback
    try:
        print("🔧 Attempting to run original bot...")
        from investment_bot import main as run_bot
        run_bot()
        print("✅ Original bot completed successfully!")
        return
    except ImportError as e:
        print(f"❌ Import error in original bot: {e}")
        print("📦 Installing dependencies...")
        install_dependencies()
        print("🔄 Retrying original bot...")
        try:
            from investment_bot import main as run_bot
            run_bot()
            print("✅ Original bot completed successfully after retry!")
            return
        except Exception as e2:
            print(f"❌ Original bot still failing: {e2}")
            traceback.print_exc()
    except Exception as e:
        print(f"❌ Error running original bot: {e}")
        traceback.print_exc()
    
    # Final fallback - simple test
    print("🔧 Running simple test as final fallback...")
    run_simple_test()

def run_simple_test():
    """Run simple test as fallback"""
    try:
        print("🧪 Running simple test...")
        from simple_test import main as run_test
        run_test()
        print("✅ Simple test completed successfully!")
    except ImportError as e:
        print(f"❌ Import error in simple test: {e}")
        print("📦 Installing dependencies...")
        install_dependencies()
        print("🔄 Retrying simple test...")
        try:
            from simple_test import main as run_test
            run_test()
            print("✅ Simple test completed successfully after retry!")
        except Exception as e2:
            print(f"❌ Simple test still failing: {e2}")
            traceback.print_exc()
            print_final_debug_info()
    except Exception as e:
        print(f"❌ Error running simple test: {e}")
        traceback.print_exc()
        print_final_debug_info()

def print_final_debug_info():
    """Print final debug information"""
    print("\n🔍 Final Debug Information:")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"📦 Python path: {sys.path}")
    print("📋 Available files:")
    os.system("ls -la")
    print("\n📦 Trying manual dependency installation...")
    install_dependencies()
    print("💡 Please check the error messages above and contact support if needed")

if __name__ == "__main__":
    main()
