#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import time
import logging
import numpy as np
from logging.handlers import RotatingFileHandler
import ccxt
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, render_template_string, request
import threading
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

# =========== Check for dependencies ==========
def check_dependencies():
    required = ['flask_httpauth', 'werkzeug', 'pandas', 'numpy', 'ccxt', 'python-dotenv']
    missing = [pkg for pkg in required if not is_installed(pkg)]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Attempting to install...")
        import subprocess
        subprocess.run(['pip', 'install'] + missing)

def is_installed(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False

# At startup
check_dependencies()

# === Logging Configuration ==================
def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            RotatingFileHandler(
                'trading_bot.log',
                maxBytes=5*1024*1024,
                backupCount=3
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# === Configuration ===
CONFIG = {
    'telegram_alerts': True,
    'min_trade_usd': 10,
    'risk_per_trade': 0.02,
    'default_take_profit': 0.05,
    'default_stop_loss': 0.03,
    'trade_cooldown': 3600,
    'commission_rate': 0.002,
    'max_volatility_sl': 0.10,
    'max_position_size': 0.1,
    'min_balance_alert': 5.00,
    'price_check_interval': 300,
    'significant_move_pct': 5.0,
    'profit_strategy': {
        'buy_threshold': 0.95,
        'sell_threshold': 1.05,
        'min_profit_pct': 3.0,
        'max_hold_days': 7
    },
    'gemini': {
        'model': 'gemini-1.5-flash',
        'max_retries': 3,
        'request_delay': 1,
        'key_cooldown': 3600
    }
}

# === API Keys ===
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "").strip()
BINGX_SECRET = os.getenv("BINGX_SECRET", "").strip()
GEMINI_KEYS = [key for key in [
    os.getenv("GEMINI_API_KEY_1", "").strip(),
    os.getenv("GEMINI_API_KEY_2", "").strip(),
    os.getenv("GEMINI_API_KEY_3", "").strip()
] if key]
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Initialize Exchange
try:
    exchange = ccxt.bingx({
        'apiKey': BINGX_API_KEY,
        'secret': BINGX_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })
    logger.info("BingX API initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize BingX API: {str(e)}")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__)

# === Global Variables ===
TOP_TOKENS = ["ETH", "BRETT", "XRP", "DOGE", "SHIB", "ACH", "TRX"]
token_data = []
data_lock = threading.Lock()
SELL_COOLDOWN = {}
balance_history = []  # Stores daily balance records
BALANCE_HISTORY_FILE = 'balance_history.json'  # File to persist balance history

# === Utility Functions ===
def safe_float(value, default=0.0):
    """Safely convert to float with validation"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def send_telegram_alert(message):
    """Send alerts to Telegram if enabled"""
    if not CONFIG['telegram_alerts'] or not TELEGRAM_BOT_TOKEN:
        return
        
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            },
            timeout=5
        )
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")

def load_balance_history():
    """Load balance history from file"""
    global balance_history
    try:
        if os.path.exists(BALANCE_HISTORY_FILE):
            with open(BALANCE_HISTORY_FILE, 'r') as f:
                balance_history = json.load(f)
                logger.info(f"Loaded balance history with {len(balance_history)} records")
    except Exception as e:
        logger.error(f"Error loading balance history: {e}")
        balance_history = []

def save_balance_history():
    """Save balance history to file"""
    try:
        with open(BALANCE_HISTORY_FILE, 'w') as f:
            json.dump(balance_history, f)
    except Exception as e:
        logger.error(f"Error saving balance history: {e}")

def record_daily_balance():
    """Record daily balance snapshot"""
    while True:
        try:
            balance = get_balances()
            if balance:
                total_usd = balance['USDT']
                for token in TOP_TOKENS:
                    if token != 'USDT' and balance.get(token, 0) > 0:
                        ticker = exchange.fetch_ticker(f"{token}/USDT")
                        total_usd += balance[token] * ticker['last']
                
                timestamp = datetime.now(timezone.utc).isoformat()
                balance_history.append({
                    'timestamp': timestamp,
                    'balance': total_usd,
                    'details': balance
                })
                
                # Keep only last 30 days
                if len(balance_history) > 30:
                    balance_history.pop(0)
                
                save_balance_history()
                logger.info(f"Recorded daily balance: ${total_usd:.2f}")
            
            # Sleep until same time tomorrow
            now = datetime.now(timezone.utc)
            tomorrow = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            sleep_seconds = (tomorrow - now).total_seconds()
            time.sleep(sleep_seconds)
            
        except Exception as e:
            logger.error(f"Error in daily balance recording: {e}")
            time.sleep(3600)  # Retry in 1 hour if error

def get_balances():
    """Fetch current account balances with validation"""
    for attempt in range(3):
        try:
            params = {'recvWindow': 5000}
            balance = exchange.fetch_balance(params)
            
            usdt_balance = safe_float(balance['total'].get('USDT', 0))
            token_balances = {
                token: safe_float(balance['total'].get(token, 0))
                for token in TOP_TOKENS
            }
            
            logger.info(f"Balances - USDT: {usdt_balance:.2f} | " +
                       " | ".join(f"{token}: {token_balances[token]:.4f}" for token in TOP_TOKENS))
            return {
                'USDT': usdt_balance,
                **token_balances
            }
            
        except Exception as e:
            if attempt == 2:
                logger.error(f"Balance fetch failed after 3 attempts: {e}")
                send_telegram_alert(f"‼️ Balance check failed: {str(e)}")
                return None
            time.sleep(1)

def calculate_indicators(df):
    """Calculate technical indicators with validation"""
    try:
        # RSI Calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        with np.errstate(divide='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['rsi'] = rsi.ffill().bfill()
        
        # EMA Calculation
        df['ema'] = df['close'].ewm(span=20, adjust=False).mean()
        
        return df
    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")
        raise

# === Trading Functions ===
def should_execute_sell(token):
    """Enhanced sell condition checker"""
    # 1. Check if we actually hold this token
    balance = get_balances()
    if not balance or balance.get(token, 0) <= 0:
        return False, "Zero balance"
    
    # 2. Verify market conditions
    try:
        ticker = exchange.fetch_ticker(f"{token}/USDT")
        if ticker['last'] is None:
            return False, "Invalid price"
    except Exception as e:
        return False, f"Ticker error: {str(e)}"
    
    # 3. Check minimum trade size
    try:
        market = exchange.market(f"{token}/USDT")
        min_cost = safe_float(market['limits']['cost']['min'], 1.0)
        position_value = ticker['last'] * balance[token]
        if position_value < min_cost:
            return False, f"Below min trade (${min_cost:.2f})"
    except Exception as e:
        return False, f"Market info error: {str(e)}"
    
    # 4. Check if in cooldown
    if token in SELL_COOLDOWN and SELL_COOLDOWN[token] > time.time():
        return False, "In cooldown"
    
    return True, "OK"

def execute_sell(token, signal):
    """Enhanced sell execution with validation"""
    try:
        can_sell, reason = should_execute_sell(token)
        if not can_sell:
            logger.warning(f"Cannot sell {token}: {reason}")
            return None
            
        symbol = f"{token}/USDT"
        ticker = exchange.fetch_ticker(symbol)
        current_price = safe_float(ticker['last'])
        
        balance = get_balances().get(token, 0)
        market = exchange.market(symbol)
        
        # Apply precision rounding
        amount = math.floor(balance / market['precision']['amount']) * market['precision']['amount']
        
        if amount <= 0:
            logger.error(f"Invalid sell amount for {token}: {amount}")
            return None
            
        logger.info(f"Executing SELL for {token}: {amount:.6f} @ {current_price:.6f}")
        
        order = exchange.create_market_sell_order(
            symbol=symbol,
            amount=amount,
            params={'recvWindow': 5000}
        )
        
        # Update cooldown
        SELL_COOLDOWN[token] = time.time() + CONFIG['trade_cooldown']
        
        send_telegram_alert(
            f"✅ *SELL EXECUTED*\n"
            f"• Token: {token}\n"
            f"• Amount: {amount:.4f}\n"
            f"• Price: ${current_price:.6f}\n"
            f"• Value: ${amount*current_price:.2f}"
        )
        return order
        
    except Exception as e:
        logger.error(f"Sell failed for {token}: {str(e)}")
        send_telegram_alert(f"⚠️ SELL FAILED {token}: {str(e)[:200]}")
        return None

def analyze_token(token):
    """Enhanced token analysis with action recommendations"""
    try:
        ohlcv = exchange.fetch_ohlcv(f"{token}/USDT", '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_indicators(df)
        
        current = df.iloc[-1]
        price_ratio = current['close'] / current['ema']
        rsi = current['rsi']
        
        # Get balance information
        balance = get_balances()
        current_position = balance.get(token, 0) if balance else 0
        
        # Generate recommendation
        recommendation = {
            'symbol': token,
            'price': current['close'],
            'ema': current['ema'],
            'rsi': rsi,
            'position': current_position,
            'value': current_position * current['close'],
            'action': 'HOLD',  # Default
            'confidence': 0,
            'reason': '',
            'can_execute': False
        }
        
        # Buy signals (only if we have no position)
        if current_position <= 0:
            if price_ratio < CONFIG['profit_strategy']['buy_threshold'] and rsi < 40:
                recommendation.update({
                    'action': 'BUY',
                    'confidence': 70,
                    'reason': 'Price below EMA and RSI oversold'
                })
        
        # Sell signals (only if we have a position)
        elif current_position > 0:
            if price_ratio > CONFIG['profit_strategy']['sell_threshold'] and rsi > 60:
                can_sell, reason = should_execute_sell(token)
                recommendation.update({
                    'action': 'SELL',
                    'confidence': 80,
                    'reason': 'Price above EMA and RSI overbought',
                    'can_execute': can_sell,
                    'execution_reason': reason if not can_sell else 'Ready to execute'
                })
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Analysis failed for {token}: {str(e)}")
        return {
            'symbol': token,
            'action': 'HOLD',
            'error': str(e)
        }

# === Data Collection Thread ===
def data_collection_thread():
    """Background thread to collect and update token data"""
    # Start balance recording thread
    balance_thread = threading.Thread(target=record_daily_balance, daemon=True)
    balance_thread.start()
    
    while True:
        try:
            new_data = []
            for token in TOP_TOKENS:
                try:
                    analysis = analyze_token(token)
                    new_data.append(analysis)
                    
                    # Execute sells immediately if recommended
                    if analysis['action'] == 'SELL' and analysis.get('can_execute', False):
                        execute_sell(token, analysis)
                    
                    time.sleep(1)  # Rate limit between tokens
                except Exception as e:
                    logger.error(f"Error processing {token}: {str(e)}")
                    continue
            
            with data_lock:
                global token_data
                token_data = new_data
            
            time.sleep(30)  # Update interval
        
        except Exception as e:
            logger.error(f"Data collection thread error: {str(e)}")
            time.sleep(60)

# === Flask Routes ===
auth = HTTPBasicAuth()

# Configure users (in production, store these securely)
users = {
    "admin": generate_password_hash(os.getenv("DASHBOARD_PASSWORD", "9026665756"))
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/')
@auth.login_required
def dashboard():
    with data_lock:
        current_data = token_data.copy()
    
    # Prepare balance chart data
    balance_dates = [datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d') 
                   for entry in balance_history]
    balance_values = [entry['balance'] for entry in balance_history]
    
    # Calculate current total balance
    current_balance = 0
    balance = get_balances()
    if balance:
        current_balance = balance['USDT']
        for token in TOP_TOKENS:
            if token != 'USDT' and balance.get(token, 0) > 0:
                ticker = exchange.fetch_ticker(f"{token}/USDT")
                current_balance += balance[token] * ticker['last']
    
    # Prepare HTML template
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/moment"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .header p {
            color: #7f8c8d;
            margin-top: 0;
        }
        .stats-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            width: 30%;
            text-align: center;
        }
        .stat-card h3 {
            margin-top: 0;
            color: #7f8c8d;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .table-container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .price-up {
            color: #27ae60;
        }
        .price-down {
            color: #e74c3c;
        }
        .action-buy {
            background-color: rgba(39, 174, 96, 0.1);
            color: #27ae60;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: 600;
        }
        .action-sell {
            background-color: rgba(231, 76, 60, 0.1);
            color: #e74c3c;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: 600;
        }
        .action-hold {
            background-color: rgba(241, 196, 15, 0.1);
            color: #f39c12;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: 600;
        }
        .last-updated {
            text-align: right;
            font-size: 0.9em;
            color: #7f8c8d;
            margin-top: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-active {
            background-color: #2ecc71;
        }
        .status-inactive {
            background-color: #e74c3c;
        }
    </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>Crypto Trading Bot Dashboard</h1>
                <p>Real-time monitoring of AI trading activities</p>
            </div>

            <div class="stats-container">
                <div class="stat-card">
                    <h3>Current Balance</h3>
                    <div class="stat-value">${{ "%.2f"|format(current_balance) }}</div>
                </div>
                <div class="stat-card">
                    <h3>24h Change</h3>
                    <div class="stat-value {% if balance_values|length > 1 and balance_values[-1] > balance_values[-2] %}price-up{% else %}price-down{% endif %}">
                        {% if balance_values|length > 1 %}
                            {{ "%.2f"|format((balance_values[-1] - balance_values[-2])/balance_values[-2]*100) }}%
                        {% else %}N/A{% endif %}
                    </div>
                </div>
                <div class="stat-card">
                    <h3>Active Positions</h3>
                    <div class="stat-value">{{ tokens|selectattr('position', 'gt', 0)|list|length }}</div>
                </div>
            </div>

            <div class="chart-container">
                <h2>Account Balance History (30 Days)</h2>
                <canvas id="balanceChart"></canvas>
            </div>

            <div class="chart-container">
                <h2>Token Price Comparison</h2>
                <canvas id="priceChart"></canvas>
            </div>

            <div class="table-container">
                <h2>Token Status Overview</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Token</th>
                            <th>Price</th>
                            <th>RSI</th>
                            <th>Position</th>
                            <th>Token Balance(USDT)</th>
                            <th>AI Recommended Action</th>
                            <th>Confidence</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for token in tokens %}
                        <tr>
                            <td>{{ token.symbol }}</td>
                            <td>{{ "%.6f"|format(token.price) }}</td>
                            <td>{{ "%.1f"|format(token.rsi) }}</td>
                            <td>{{ "%.4f"|format(token.position) }}</td>
                            <td>{{ "%.2f"|format(token.value) }}</td>
                            <td class="action-{{ token.action.lower() }}">{{ token.action }}</td>
                            <td>{{ token.confidence }}</td>
                            <td>
                                {% if token.action == 'SELL' %}
                                    {% if token.can_execute %}
                                        ✅ Ready
                                    {% else %}
                                        ❌ {{ token.execution_reason }}
                                    {% endif %}
                                {% else %}
                                    {{ token.reason }}
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                <div class="last-updated">
                    <p>Last updated: {{ now }} <b>UTC</b></p>
                </div>
            </div>
        </div>

        <script>
            // Balance Chart
            function createBalanceChart() {
                const ctx = document.getElementById('balanceChart').getContext('2d');
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: {{ balance_dates|tojson }},
                        datasets: [{
                            label: 'Total Account Balance (USD)',
                            data: {{ balance_values|tojson }},
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 2,
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                title: {
                                    display: true,
                                    text: 'Balance (USD)'
                                }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `Balance: $${context.raw.toFixed(2)}`;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            // Price Chart
            function createPriceChart() {
                const ctx = document.getElementById('priceChart').getContext('2d');
                
                // Prepare data
                const labels = {{ tokens|map(attribute='symbol')|list|tojson }};
                const prices = {{ tokens|map(attribute='price')|list|tojson }};
                const actions = {{ tokens|map(attribute='action')|list|tojson }};
                
                const backgroundColors = actions.map(action => 
                    action === 'BUY' ? 'rgba(39, 174, 96, 0.7)' : 
                    action === 'SELL' ? 'rgba(231, 76, 60, 0.7)' : 
                    'rgba(241, 196, 15, 0.7)'
                );
                
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Current Price (USDT)',
                            data: prices,
                            backgroundColor: backgroundColors,
                            borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                title: {
                                    display: true,
                                    text: 'Price (USDT)'
                                }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const token = {{ tokens|tojson }}[context.dataIndex];
                                        return [
                                            `Price: ${token.price.toLocaleString(undefined, { minimumFractionDigits: token.price < 1 ? 6 : 2, maximumFractionDigits: token.price < 1 ? 6 : 2 })} USDT`,
                                            `AI Action: ${token.action}`,
                                            `RSI: ${token.rsi.toFixed(1)}`,
                                            `Position: ${token.position.toFixed(4)} ${token.symbol}`
                                        ];
                                    }
                                }
                            }
                        }
                    }
                });
            }

            // Initialize the dashboard
            document.addEventListener('DOMContentLoaded', function() {
                createBalanceChart();
                createPriceChart();
                
                // Auto-refresh every 30 seconds
                setInterval(function() {
                    fetch(window.location.href)
                        .then(response => response.text())
                        .then(html => {
                            const parser = new DOMParser();
                            const newDoc = parser.parseFromString(html, 'text/html');
                            document.querySelector('.table-container').innerHTML = 
                                newDoc.querySelector('.table-container').innerHTML;
                            document.querySelector('.stats-container').innerHTML = 
                                newDoc.querySelector('.stats-container').innerHTML;
                        })
                        .catch(error => console.error('Error refreshing data:', error));
                }, 30000);
            });
        </script>
    </body>
    </html>
    """, 
    tokens=current_data,
    now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    balance_dates=balance_dates,
    balance_values=balance_values,
    current_balance=current_balance)

@app.route('/force_sell/<token>')
@auth.login_required
def force_sell(token):
    if token not in TOP_TOKENS:
        return "Invalid token", 400
        
    # Bypass normal checks for testing
    try:
        symbol = f"{token}/USDT"
        balance = get_balances().get(token, 0)
        if balance <= 0:
            return f"Cannot sell - zero balance of {token}", 400
            
        market = exchange.market(symbol)
        amount = math.floor(balance / market['precision']['amount']) * market['precision']['amount']
        
        order = exchange.create_market_sell_order(
            symbol=symbol,
            amount=amount,
            params={'recvWindow': 5000}
        )
        
        SELL_COOLDOWN[token] = time.time() + CONFIG['trade_cooldown']
        return f"Force sell executed: {order['id']}", 200
        
    except Exception as e:
        return f"Force sell failed: {str(e)}", 500

# === Main Execution ===
if __name__ == '__main__':
    # Load balance history
    load_balance_history()
    
    # Start data collection thread
    thread = threading.Thread(target=data_collection_thread, daemon=True)
    thread.start()
    
    # Start Flask server
    app.run(host="0.0.0.0", port=8080)