#!/usr/bin/env python3
"""
Keep Alive script for Replit
Prevents the repl from going to sleep
"""

from flask import Flask
from threading import Thread
import time
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return "Investment Bot is running! 🚀"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    """Keep the repl alive by pinging it"""
    while True:
        try:
            # Ping your own repl
            requests.get('https://your-repl-name.your-username.repl.co')
            print("🔄 Pinged to keep alive")
        except:
            pass
        time.sleep(300)  # Ping every 5 minutes

def start_server():
    """Start the Flask server"""
    server = Thread(target=run)
    server.start()
    
    # Start keep alive in background
    keep_alive_thread = Thread(target=keep_alive)
    keep_alive_thread.daemon = True
    keep_alive_thread.start()

if __name__ == "__main__":
    start_server()
