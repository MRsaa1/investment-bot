#!/usr/bin/env python3
"""
Keep Alive script for Replit
Prevents the repl from going to sleep
Updated for better reliability
"""

import os
import time
import threading
from datetime import datetime

try:
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not available, using simple keep alive")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ Requests not available, using simple keep alive")

if FLASK_AVAILABLE:
    app = Flask(__name__)

    @app.route('/')
    def home():
        return f"""
        <html>
        <head>
            <title>Investment Bot</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                .status {{ color: green; font-size: 24px; }}
                .time {{ color: gray; font-size: 16px; }}
            </style>
        </head>
        <body>
            <h1>🚀 Investment Bot</h1>
            <div class="status">✅ Running Successfully</div>
            <div class="time">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <p>This bot generates investment analysis reports for stocks.</p>
        </body>
        </html>
        """

    @app.route('/health')
    def health():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    def run_flask():
        """Run Flask server"""
        try:
            app.run(host='0.0.0.0', port=8080)
        except Exception as e:
            print(f"❌ Flask server error: {e}")

def keep_alive_simple():
    """Simple keep alive without external dependencies"""
    while True:
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"🔄 Keep alive ping at {current_time}")
            
            # Create a simple file to show activity
            with open('keep_alive.log', 'a') as f:
                f.write(f"{current_time} - Bot is alive\n")
                
        except Exception as e:
            print(f"❌ Keep alive error: {e}")
        
        time.sleep(300)  # Sleep for 5 minutes

def keep_alive_with_requests():
    """Keep alive with requests (if available)"""
    repl_url = os.environ.get('REPLIT_URL', '')
    if not repl_url:
        print("⚠️ REPLIT_URL not set, using simple keep alive")
        keep_alive_simple()
        return
    
    while True:
        try:
            response = requests.get(f"{repl_url}/health", timeout=10)
            if response.status_code == 200:
                print("🔄 Successfully pinged repl")
            else:
                print(f"⚠️ Ping returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Ping failed: {e}")
            print("🔄 Falling back to simple keep alive")
            keep_alive_simple()
            break
        
        time.sleep(300)  # Sleep for 5 minutes

def start_server():
    """Start the keep alive system"""
    print("🚀 Starting keep alive system...")
    
    if FLASK_AVAILABLE:
        # Start Flask server in background
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("✅ Flask server started")
        
        # Start keep alive
        if REQUESTS_AVAILABLE:
            keep_alive_thread = threading.Thread(target=keep_alive_with_requests, daemon=True)
        else:
            keep_alive_thread = threading.Thread(target=keep_alive_simple, daemon=True)
        
        keep_alive_thread.start()
        print("✅ Keep alive thread started")
    else:
        # Fallback to simple keep alive
        keep_alive_thread = threading.Thread(target=keep_alive_simple, daemon=True)
        keep_alive_thread.start()
        print("✅ Simple keep alive started")

if __name__ == "__main__":
    start_server()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Keep alive stopped by user")
