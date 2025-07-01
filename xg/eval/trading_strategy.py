import json
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import numpy as np  # type: ignore
import os
from typing import Callable, List, Tuple
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt #type: ignore
import base64
from io import BytesIO
from matplotlib import rcParams  #type: ignore
import matplotlib.dates as mdates  #type: ignore
import matplotlib.ticker as ticker  #type: ignore
from matplotlib.ticker import MaxNLocator #type: ignore
import requests
from textblob import TextBlob  # pip install textblob
import multiprocessing

YOUR_SECRET = "81zHs9ZUNsP2o2HL8grxv7oBVjBsUgcviSCrMese"
YOUR_KEY = "PKMI001E6DXX6L93PF92"

def process_stock(args):
    symbol, groups, json_path = args

    results = []
    cumulative = {}
    total_initial_buy_hold = 0.0
    total_profit_buy_hold = 0.0

    # You need to create a StrategyTester instance INSIDE the worker if you need class methods!
    tester = StrategyTester(json_path)  # Or pass the file path as an argument

    for group_index, group in enumerate(groups):
        if not group:
            continue
        results_with_plots = tester.evaluate_all_combinations_group(group, symbol, group_index)
        results.append((group_index, results_with_plots, group))        
        shares_bought = 100.0 / group[0].open if group[0].open != 0 else 0.0
        total_initial_buy_hold += 100.0
        total_profit_buy_hold += (group[-1].close - group[0].open) * shares_bought

    # Return final_decisions and final_positions for this stock
    final_decisions = getattr(tester, 'final_decisions', {})
    final_positions = getattr(tester, 'final_positions', {})
    return symbol, results, total_initial_buy_hold, total_profit_buy_hold, cumulative, groups, final_decisions, final_positions

# Move this OUTSIDE the StrategyTester class, at module level (around line 50)
def process_stock_rolling(args):
    symbol, groups, json_path, use_rolling_windows = args
    
    results = []
    cumulative = {}
    total_initial_buy_hold = 0.0
    total_profit_buy_hold = 0.0

    tester = StrategyTester(json_path)
    tester.use_rolling_windows = use_rolling_windows

    for group_index, group in enumerate(groups):
        if not group:
            continue
            
        if use_rolling_windows:
            # ✅ USE ROLLING WINDOW APPROACH
            rolling_results = tester.rolling_window_backtest_group(group, symbol, group_index)
            
            if rolling_results:
                # Calculate overall performance from rolling windows
                total_profit, total_trades, correct_trades, execution_log = tester.calculate_rolling_window_performance(rolling_results, group)
                
                # Get the final (most recent) strategy decision
                final_rolling_result = rolling_results[-1]
                _, best_strategy, final_decision, trade_signals, final_position, l5_method = final_rolling_result
                
                # ✅ FIX: Use the actual best strategy's method name instead of generic name
                rolling_performance = PredictionResult(
                    method=best_strategy.method,  # Use actual method like "pred_1_1"
                    level1_method=best_strategy.level1_method,  # Use actual L1 method
                    level2_method=best_strategy.level2_method,  # Use actual L2 method
                    level3_method=best_strategy.level3_method,  # Use actual L3 method
                    level4_method=best_strategy.level4_method,  # Use actual L4 method
                    profit=total_profit,
                    total_trades=total_trades,
                    total_buys=total_trades,
                    correct_trades=correct_trades,
                    mean_prediction_error=0.0,
                    level5_method=best_strategy.level5_method  # Use actual L5 method
                )
                
                results.append((group_index, [(rolling_performance, final_decision, trade_signals, final_position, "Rolling")], group, rolling_results))
            else:
                # If no rolling results, still append with None
                results.append((group_index, [], group, None))
        else:
            
            results_with_plots = tester.evaluate_all_combinations_group(group, symbol, group_index)
            results.append((group_index, results_with_plots, group, None))
            
        # Calculate buy & hold for comparison
        shares_bought = 100.0 / group[0].open if group[0].open != 0 else 0.0
        total_initial_buy_hold += 100.0
        total_profit_buy_hold += (group[-1].close - group[0].open) * shares_bought

    # Return results compatible with existing system
    final_decisions = getattr(tester, 'final_decisions', {})
    final_positions = getattr(tester, 'final_positions', {})
    return symbol, results, total_initial_buy_hold, total_profit_buy_hold, cumulative, groups, final_decisions, final_positions

def level4_linear(score: float) -> float:
    """Linearly scales between -1 and 1"""
    return max(-1.0, min(1.0, score))

def level4_threshold(score: float, threshold: float = 0.3) -> float:
    if score >= threshold:
        return 1.0
    elif score <= -threshold:
        return -1.0
    return 0.0

def level4_capped(score: float, cap: float = 0.5) -> float:
    if score > 0:
        return min(score, cap)
    else:
        return max(score, -cap)

def level4_sigmoid(score: float, scale: float = 1.0) -> float:
    """Sigmoid scaling from -1 to 1"""
    return 2 / (1 + math.exp(-scale * score)) - 1

# Wrapper for all confidence sizing methods
confidence_based_level4_methods = {
    'linear': level4_linear,
    'threshold': level4_threshold,
    'capped': level4_capped,
    'sigmoid': level4_sigmoid
}

def apply_confidence_sizing(score: float, method: Callable[[float], float]) -> float:
    """Applies confidence-based Level 4 method"""
    return method(score)

# ✅ Model-confidence blending (new Level 1)
def level1_confidence_blend(pred1: List[float], pred2: List[float], error1: float, error2: float) -> List[float]:
    total_error = error1 + error2
    if total_error == 0:
        weight1 = 0.5
    else:
        weight1 = error2 / total_error  # lower error → higher weight
    weight2 = 1 - weight1
    return [p1 * weight1 + p2 * weight2 for p1, p2 in zip(pred1, pred2)]

# ✅ Error-based forecast adjustment (new Level 1)
def level1_error_adjusted(predictions: List[float], recent_errors: List[float]) -> List[float]:
    if not recent_errors:
        return predictions
    avg_bias = sum(recent_errors) / len(recent_errors)
    return [p + avg_bias for p in predictions]

# ✅ Market Regime Detection

def detect_market_regime(prices: List[float], window: int = 5, threshold: float = 1.0) -> str:
    """
    Simple regime detection:
    - Bull: upward trend
    - Bear: downward trend
    - Sideways: neither
    """
    if len(prices) < window:
        return "unknown"
    trend = prices[-1] - prices[-window]
    if trend > threshold:
        return "bull"
    elif trend < -threshold:
        return "bear"
    else:
        return "sideways"

# ✅ Perfect Trading Strategy (Baseline Comparison)
def perfect_trade_profit(prices: List[float]) -> Tuple[float, int]:
    """
    Simulates perfect trades — buy before every rise, sell before every fall.
    Uses open prices, assumes perfect foresight.
    Returns total profit and number of trades made.
    """
    profit = 0.0
    trades = 0
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            profit += diff
            trades += 1
    return profit, trades

# ✅ Yesterday Trend-Following Strategy (Baseline Comparison)
def trend_following_strategy(prices: List[float]) -> Tuple[float, int, float]:
    """
    Buys if yesterday was an up day, sells if yesterday was down.
    Simplified: tracks position and exit at next open.
    Returns profit, trade count, and win rate.
    """
    profit = 0.0
    position = 0.0
    trade_count = 0
    win_count = 0
    for i in range(2, len(prices)):
        yesterday_trend = prices[i - 1] - prices[i - 2]
        if position == 0.0:
            if yesterday_trend > 0:
                position = prices[i]  # buy at today's open
        else:
            if yesterday_trend < 0:
                result = prices[i] - position  # sell
                profit += result
                if result > 0:
                    win_count += 1
                trade_count += 1
                position = 0.0
    win_rate = (win_count / trade_count) if trade_count > 0 else 0.0
    return profit, trade_count, win_rate

# 🎨 Sleek HTML CSS Styles
def get_sleek_html_css() -> str:
    return """
    <style>
      body {
        font-family: 'Segoe UI', sans-serif;
        background: #f9fbfd;
        color: #333;
        margin: 0;
        padding: 20px;
      }
      .container {
        max-width: 1000px;
        margin: auto;
      }
      .stock-section {
        background: #fff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px;
      }
      .stock-section h2 {
        margin-top: 0;
        color: #2c3e50;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
      }
      th, td {
        border: 1px solid #ccc;
        padding: 10px;
        text-align: center;
      }
      th {
        background-color: #eef3f7;
        color: #2c3e50;
      }
      .strategy-label {
        font-weight: bold;
        color: #34495e;
      }
      .comparison-row {
        background-color: #fefefe;
      }
    </style>
    """


def generate_sortable_html_table(group_index, symbol, results, group_initial):
    table_id = f"table_{symbol}_{group_index}"
    search_id = f"search_{symbol}_{group_index}"
    
    html = f"""
    <h4>All Tested Strategies</h4>
    <input type="text" id="{search_id}" onkeyup="searchTable('{search_id}', '{table_id}')" placeholder="Search strategies..." class="search-box">
    <table id="{table_id}" data-sort-dir="asc">
        <tr>
            
            <th onclick="sortTable('{table_id}', 0)">Prediction</th>
            <th onclick="sortTable('{table_id}', 1)">L1</th>
            <th onclick="sortTable('{table_id}', 2)">L2</th>
            <th onclick="sortTable('{table_id}', 3)">L3</th>
            <th onclick="sortTable('{table_id}', 4)">L4</th>
            <th onclick="sortTable('{table_id}', 5)">L5</th>
            <th onclick="sortTable('{table_id}', 6)">Profit ($)</th>
            <th onclick="sortTable('{table_id}', 7)">% Profit</th>
            <th onclick="sortTable('{table_id}', 8)">Total Trades</th>
            <th onclick="sortTable('{table_id}', 9)">Accuracy (%)</th>
            <th onclick="sortTable('{table_id}', 10)">Mean Error</th>
                        
        </tr>
    """
    
    # ✅ MOVE THE RETURN STATEMENT TO THE END
    # return html  # ❌ This returns too early!
    
    # ✅ ADD THE MISSING LOOP HERE
    for res_tuple in results:
        # Support both (PredictionResult, ...) and (PredictionResult, ..., l5_method_name)
        if len(res_tuple) == 5:
            res, _, _, _, l5_method_name = res_tuple
        else:
            res, _, _, _ = res_tuple
            l5_method_name = getattr(res, "level5_method", "L5_none")
        pct_profit = (res.profit / 100.0 * 100)
        accuracy = (res.correct_trades / res.total_trades * 100) if res.total_trades > 0 else 0
        html += f"""
        <tr>
            <td>{res.method}</td>
            <td>{res.level1_method}</td>
            <td>{res.level2_method}</td>
            <td>{res.level3_method}</td>
            <td>{res.level4_method}</td>
            <td>{l5_method_name}</td>
            <td>${res.profit:.2f}</td>
            <td>{pct_profit:.2f}%</td>
            <td>{res.total_trades}</td>
            <td>{accuracy:.2f}%</td>
            <td>${res.mean_prediction_error:.2f}</td>
        </tr>
        """
    
    # ✅ MOVE THE CLOSING HTML AND RETURN HERE
    html += """
    </table>
    <script>
    function sortTable(tableId, n) {
        // Add your sorting JavaScript here if needed
    }
    function searchTable(searchId, tableId) {
        // Add your search JavaScript here if needed
    }
    </script>
    """
    
    return html

@dataclass
class TradingDay:
    date: str
    open: float
    close: float
    rsi: float
    williams: float
    adx: float
    pred_1_1: List[float]
    pred_1_2: List[float]
    pred_2_1: List[float]
    pred_2_2: List[float]
    pred_3_1: List[float]
    pred_3_2: List[float]
    pred_4_1: List[float]
    pred_4_2: List[float]
    pred_5_1: List[float]
    pred_5_2: List[float]
    pred_6_1: List[float]
    pred_6_2: List[float]
    pred_1_3: List[float]
    pred_2_3: List[float]
    pred_3_3: List[float]
    pred_4_3: List[float]
    pred_5_3: List[float]
    pred_6_3: List[float]

@dataclass
class PredictionResult:
    method: str
    level1_method: str
    level2_method: str
    level3_method: str
    level4_method: str
    profit: float
    total_trades: int
    total_buys: int
    correct_trades: int
    mean_prediction_error: float
    level5_method: str = ""

    def __str__(self):
        accuracy = (self.correct_trades / self.total_buys * 100) if self.total_buys > 0 else 0
        return (f"Method: {self.method}\n"
                f"Level 1: {self.level1_method}\n"
                f"Level 2: {self.level2_method}\n"
                f"Level 3: {self.level3_method}\n"
                f"Level 4: {self.level4_method}\n"
                f"Profit: ${self.profit:.2f}\n"
                f"Total Trades: {self.total_trades}\n"
                f"Correct Trades: {self.correct_trades}\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"Mean Prediction Error: ${self.mean_prediction_error:.2f}")

@dataclass
class WeightedPrediction:
    weight: float
    error: float  # Mean absolute error of the weighted prediction
    error1: float = 0.0  # Error of pred1
    error2: float = 0.0  # Error of pred2

def generate_inline_plot(group: List[TradingDay], trade_signals: List[Tuple[str, float]], symbol: str, group_index: int) -> str:
    # Prepare price timeline with spacing
    prices = []
    x_vals = []
    dates_for_xticks = []
    tick_positions = []
    spacing = 0.25
    current_x = 0

    opens = []
    closes = []
    open_x = []
    close_x = []

    for i, day in enumerate(group):
        prices.extend([day.open, day.close])
        x_vals.extend([current_x, current_x + spacing])
        dates_for_xticks.append(day.date)
        tick_positions.append(current_x + spacing / 2)
        # For open/close lines
        opens.append(day.open)
        closes.append(day.close)
        open_x.append(current_x)
        close_x.append(current_x + spacing)
        current_x += 1  # larger spacing between days

    # Align plotted trades with backtest: buy/sell at i+1 open (historical trades - triangles)
    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    holding = False
    for i, (_, signal) in enumerate(trade_signals):
        if signal == 1 and not holding and i+1 < len(group):
            buys_x.append(open_x[i+1])
            buys_y.append(opens[i+1])
            holding = True
        elif signal == -1 and holding and i+1 < len(group):
            sells_x.append(open_x[i+1])
            sells_y.append(opens[i+1])
            holding = False

    # ✅ ADD FUTURE SIGNALS (squares)
    future_buys_x, future_buys_y, future_sells_x, future_sells_y = [], [], [], []
    
    # Get the final signal from the last day (this would be the next trading decision)
    if trade_signals:
        final_signal = trade_signals[-1][1]  # Last signal from the strategy
        last_day = group[-1]
        
        # Position the future signal at the end of the chart
        future_x = open_x[-1] + 1  # One position beyond the last day
        future_price = last_day.close  # Use last close price as reference
        
        if final_signal == 1:  # Buy signal
            future_buys_x.append(future_x)
            future_buys_y.append(future_price)
        elif final_signal == -1:  # Sell signal
            future_sells_x.append(future_x)
            future_sells_y.append(future_price)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Main open→close line
    ax.plot(x_vals, prices, color='black', linewidth=2, label='Open→Close Line')
    # Add open and close lines
    ax.plot(open_x, opens, color='#8ecae6', linestyle='--', linewidth=1, alpha=0.7, label='Open Price')
    ax.plot(close_x, closes, color='#ffb703', linestyle='--', linewidth=1, alpha=0.7, label='Close Price')
    
    # Historical buy/sell markers (triangles)
    ax.scatter(buys_x, buys_y, color='green', marker='^', s=120, label='Historical Buy')
    ax.scatter(sells_x, sells_y, color='red', marker='v', s=120, label='Historical Sell')
    
    # ✅ Future buy/sell markers (squares)
    if future_buys_x:
        ax.scatter(future_buys_x, future_buys_y, color='darkgreen', marker='s', s=150, label='Future Buy Signal', edgecolors='black', linewidth=2)
    if future_sells_x:
        ax.scatter(future_sells_x, future_sells_y, color='darkred', marker='s', s=150, label='Future Sell Signal', edgecolors='black', linewidth=2)
    
    # ✅ Add a vertical line to separate historical from future
    if future_buys_x or future_sells_x:
        ax.axvline(x=open_x[-1] + 0.5, color='gray', linestyle=':', alpha=0.7, label='Future Signal')
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(dates_for_xticks, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()

    buf = BytesIO() 
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{encoded}" alt="Signal Chart for {symbol} Group {group_index}" style="width:100%;max-width:800px;margin-top:10px;">'

def generate_rolling_window_plot(group: List[TradingDay], rolling_results: List[Tuple], symbol: str, group_index: int) -> str:
    """Generate plot showing rolling window trading signals with future signal"""
    # Prepare price timeline with spacing (same as original)
    prices = []
    x_vals = []
    dates_for_xticks = []
    tick_positions = []
    spacing = 0.25
    current_x = 0

    opens = []
    closes = []
    open_x = []
    close_x = []

    for i, day in enumerate(group):
        prices.extend([day.open, day.close])
        x_vals.extend([current_x, current_x + spacing])
        dates_for_xticks.append(day.date)
        tick_positions.append(current_x + spacing / 2)
        # For open/close lines
        opens.append(day.open)
        closes.append(day.close)
        open_x.append(current_x)
        close_x.append(current_x + spacing)
        current_x += 1  # larger spacing between days

    # ✅ ROLLING WINDOW TRADE MARKERS
    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    position = 0  # Track position to align with actual trading logic
    
    for signal_day, best_strategy, final_decision, trade_signals, final_position, l5_method in rolling_results:
        if signal_day + 1 < len(group):  # Execute on next day
            execution_day = group[signal_day + 1]
            execution_x = open_x[signal_day + 1]
            execution_price = opens[signal_day + 1]
            
            if final_decision == 1 and position == 0:  # Buy signal
                buys_x.append(execution_x)
                buys_y.append(execution_price)
                position = 1
            elif final_decision == -1 and position == 1:  # Sell signal
                sells_x.append(execution_x)
                sells_y.append(execution_price)
                position = 0

    # ✅ FUTURE SIGNAL (from last rolling result)
    future_buys_x, future_buys_y, future_sells_x, future_sells_y = [], [], [], []
    
    if rolling_results:
        # Get the most recent rolling result (this is the future signal)
        final_rolling_result = rolling_results[-1]
        _, _, future_signal, _, _, _ = final_rolling_result
        
        # Position the future signal at the end of the chart
        future_x = open_x[-1] + 1  # One position beyond the last day
        future_price = group[-1].close  # Use last close price as reference
        
        if future_signal == 1:  # Future buy signal
            future_buys_x.append(future_x)
            future_buys_y.append(future_price)
        elif future_signal == -1:  # Future sell signal
            future_sells_x.append(future_x)
            future_sells_y.append(future_price)

    # ✅ CREATE THE PLOT
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Main open→close line
    ax.plot(x_vals, prices, color='black', linewidth=2, label='Open→Close Line')
    
    # Add open and close lines
    ax.plot(open_x, opens, color='#8ecae6', linestyle='--', linewidth=1, alpha=0.7, label='Open Price')
    ax.plot(close_x, closes, color='#ffb703', linestyle='--', linewidth=1, alpha=0.7, label='Close Price')
    
    # Historical rolling window buy/sell markers (triangles)
    if buys_x:
        ax.scatter(buys_x, buys_y, color='green', marker='^', s=120, label='Rolling Buy', alpha=0.8)
    if sells_x:
        ax.scatter(sells_x, sells_y, color='red', marker='v', s=120, label='Rolling Sell', alpha=0.8)
    
    # ✅ Future buy/sell markers (squares) - BIGGER AND MORE PROMINENT
    if future_buys_x:
        ax.scatter(future_buys_x, future_buys_y, color='darkgreen', marker='s', s=200, 
                  label='Future Buy Signal', edgecolors='black', linewidth=3, alpha=0.9)
    if future_sells_x:
        ax.scatter(future_sells_x, future_sells_y, color='darkred', marker='s', s=200, 
                  label='Future Sell Signal', edgecolors='black', linewidth=3, alpha=0.9)
    
    # ✅ Add a vertical line to separate historical from future
    if future_buys_x or future_sells_x:
        ax.axvline(x=open_x[-1] + 0.5, color='gray', linestyle=':', linewidth=2, alpha=0.8, label='Future Signal Line')
    
    # ✅ Add rolling window indicator
    ax.text(0.02, 0.98, f'Rolling {len(rolling_results)}-Day Adaptive Strategy', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Formatting
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(dates_for_xticks, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f'{symbol} - Rolling Window Strategy (Group {group_index})', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()

    # Save to base64
    buf = BytesIO() 
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    
    return f'<img src="data:image/png;base64,{encoded}" alt="Rolling Window Chart for {symbol} Group {group_index}" style="width:100%;max-width:800px;margin-top:10px;border:2px solid #ddd;border-radius:8px;">'

class StrategyTester:
    def __init__(self, json_file: str, max_group_length: int = 20):
        # Ensure optimal_weights is defined before loading data
        self.optimal_weights = {}  # symbol -> list of optimal weights dict per group
        self.trading_data = self._load_data(json_file, max_group_length)
        # We'll store per-stock and global buy-and-hold metrics here.
        self.buy_hold_by_stock = {}  # symbol -> (total_profit, percent_profit)
        self.global_buy_hold = None
        
        # ✅ ADD SIMPLE SLIPPAGE CONFIGURATION
        self.slippage_config = {
            'base_slippage_bps': 0,  # 12.5 basis points (0.125%) base slippage
        }

        # ✅ ADD ROLLING WINDOW CONFIGURATION
        self.rolling_window_size = 10
        self.use_rolling_windows = True  # Set to False to use original fixed groups
        
        # ✅ ADD MISSING ATTRIBUTE
        self.table = False  # Add this line to control table display
        
    def calculate_slippage_adjusted_price(self, target_open_price: float, decision: int) -> float:
        """
        Calculate slippage-adjusted execution price - ALWAYS EXECUTES.
        
        Args:
            target_open_price: The open price we want to trade at
            decision: 1 for buy, -1 for sell
            
        Returns:
            Slippage-adjusted execution price
        """
        # ✅ SIMPLE FIXED SLIPPAGE
        base_slippage = self.slippage_config['base_slippage_bps'] / 10000  # Convert bps to decimal
        
        if decision == 1:  # Buy order - PAY MORE
            execution_price = target_open_price * (1 + base_slippage)
        elif decision == -1:  # Sell order - RECEIVE LESS
            execution_price = target_open_price * (1 - base_slippage)
        elif decision == 0:  # Hold - no slippage needed
            execution_price = target_open_price
        else:
            raise ValueError(f"Invalid decision value: {decision}. Expected 1 (buy), -1 (sell), or 0 (hold)")
        
        return execution_price

    def _load_data(self, json_file: str, max_group_length: int) -> Dict[str, List[List[TradingDay]]]:
        """
        Loads data from the JSON file.
        Expected JSON structure:
           { "AAPL": [ { "data": [ {entry}, {entry}, ... ] }, { "data": [...] } ],
             "GOOG": [ { "data": [...] }, ... ]
           }
        For each entry, every prediction field (keys starting with 'pred_')
        is a comma‐separated string. The first numerical value from each list is removed.
        """
        with open(json_file, 'r') as f:
            raw_data = json.load(f)

        processed_data: Dict[str, List[List[TradingDay]]] = {}
        for symbol, groups in raw_data.items():
            processed_groups = []
            self.optimal_weights[symbol] = []
            for group in groups:
                group_data = group['data'][-max_group_length:]
                trading_days = []
                group_optimal_weights = {}
                # First pass: calculate optimal weights per prediction pair for this group
                for i, entry in enumerate(group_data):
                    predictions = {}
                    for key, value in entry.items():
                        if key.startswith('pred_'):
                            float_list = [float(x.strip()) for x in value.strip('[]').split(',')]
                            #float_list = [float(x.strip()) for x in value.split(',')]
                            float_list = float_list[1:]
                            predictions[key] = float_list
                    if i < len(group_data) - max(len(predictions.get('pred_1_1', [])), 1):
                        future_prices = [float(group_data[i + j + 1]['close'])
                                         for j in range(len(predictions.get('pred_1_1', [])))]
                        for pred_num in range(1, 7):
                            pred1_key = f'pred_{pred_num}_1'
                            pred2_key = f'pred_{pred_num}_2'
                            if pred1_key in predictions and pred2_key in predictions:
                                weight_info = self.calculate_optimal_weight(
                                    predictions[pred1_key],
                                    predictions[pred2_key],
                                    future_prices
                                )
                                if pred_num not in group_optimal_weights:
                                    group_optimal_weights[pred_num] = weight_info
                # Second pass: create TradingDay objects using the computed optimal weights
                for entry in group_data:
                    predictions = {}
                    for key, value in entry.items():
                        if key.startswith('pred_'):
                            float_list = [float(x.strip()) for x in value.strip('[]').split(',')]
                            #float_list = [float(x.strip()) for x in value.split(',')]
                            #float_list = float_list[1:]
                            predictions[key] = float_list
                    for pred_num in range(1, 7):
                        weight = group_optimal_weights.get(pred_num, WeightedPrediction(0.5, 0.0)).weight
                        pred1_key = f'pred_{pred_num}_1'
                        pred2_key = f'pred_{pred_num}_2'
                        pred3_key = f'pred_{pred_num}_3'
                        if pred1_key in predictions and pred2_key in predictions:
                            predictions[pred3_key] = [
                                weight * p1 + (1 - weight) * p2
                                for p1, p2 in zip(predictions[pred1_key], predictions[pred2_key])
                            ]
                    trading_day = TradingDay(
                        date=entry['date'],
                        open=float(entry['open']),
                        close=float(entry['close']),
                        rsi=float(entry['rsi']),
                        williams=float(entry['williams']),
                        adx=float(entry['adx']),
                        **predictions
                    )
                    trading_days.append(trading_day)
                processed_groups.append(trading_days)
                self.optimal_weights[symbol].append(group_optimal_weights)
            processed_data[symbol] = processed_groups
        return processed_data

    # Level 1 methods
    def level1_raw(self, predictions: List[float]) -> List[float]:
        return predictions[:3]

    def level1_simple_average(self, predictions: List[float]) -> List[float]:
        result = []
        for i in range(3):
            result.append(np.mean(predictions[i:i+5]))
        return result

    def level1_weighted_average(self, predictions: List[float], alpha: float = 0.5) -> List[float]:
        result = []
        for i in range(3):
            subset = predictions[i:i+5]
            n = len(subset)
            weights = [alpha * (1 - alpha) ** (n - 1 - j) for j in range(n)]
            total_weight = sum(weights)
            result.append(sum(p * w for p, w in zip(subset, weights)) / total_weight)
        return result

    def level1_exponential_average(self, predictions: List[float], alpha: float = 0.3) -> List[float]:
        result = []
        for i in range(3):
            subset = predictions[i:i+5]
            weights = [alpha * (1 - alpha) ** (len(subset)-1-j) for j in range(len(subset))]
            weights = [w/sum(weights) for w in weights]
            result.append(sum(p * w for p, w in zip(subset, weights)))
        return result

    def level1_median_filter(self, predictions: List[float]) -> List[float]:
        result = []
        for i in range(3):
            result.append(np.median(predictions[i:i+5]))
        return result

    # Level 2 methods
    def level2_simple_comparison(self, adjusted_values: List[float], close: float) -> int:
        return 1 if adjusted_values[0] > close else -1

    def level2_trend_analysis(self, adjusted_values: List[float], close: float) -> int:
        if all(adjusted_values[i] < adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return 1
        if all(adjusted_values[i] > adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return -1
        return 0

    def level2_threshold(self, adjusted_values: List[float], close: float) -> int:
        percent_change = (adjusted_values[0] - close) / close * 100
        if percent_change >= 2:
            return 1
        if percent_change <= -2:
            return -1
        return 0

    def level2_combined(self, adjusted_values: List[float], close: float) -> int:
        percent_change = (adjusted_values[0] - close) / close * 100
        if percent_change >= 2 and all(adjusted_values[i] < adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return 1
        if percent_change <= -2 and all(adjusted_values[i] > adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return -1
        return 0

    def level2_momentum(self, adjusted_values: List[float], close: float) -> int:
        momentum = (adjusted_values[0] - adjusted_values[-1]) / adjusted_values[-1] * 100
        if momentum > 1.5:
            return 1
        if momentum < -1.5:
            return -1
        return 0

    def level2_volatility_based(self, adjusted_values: List[float], close: float) -> int:
        volatility = np.std(adjusted_values) / np.mean(adjusted_values) * 100
        price_change = (adjusted_values[0] - close) / close * 100
        if volatility < 1.0:
            if price_change > 1.0:
                return 1
            if price_change < -1.0:
                return -1
        else:
            if price_change > 2.0:
                return 1
            if price_change < -2.0:
                return -1
        return 0
    
    # Level 3 methods
    def level3_no_adjustment(self, day: TradingDay) -> float:
        return 0.0

    def level3_rsi(self, day: TradingDay) -> float:
        rsi_factor = max(-0.5, min(0.5, (30 - day.rsi) / 100))
        return rsi_factor

    def level3_williams(self, day: TradingDay) -> float:
        williams_factor = max(-0.5, min(0.5, (-80 - day.williams) / 100))
        return williams_factor

    def level3_combined(self, day: TradingDay) -> float:
        rsi_factor = max(-0.5, min(0.5, (30 - day.rsi) / 100))
        williams_factor = max(-0.5, min(0.5, (-80 - day.williams) / 100))
        return (rsi_factor * 0.5 + williams_factor * 0.5)

    def level3_adx(self, day: TradingDay) -> float:
        adx_factor = max(-0.5, min(0.5, (day.adx - 25) / 100))
        return adx_factor

    def level3_all_combined(self, day: TradingDay) -> float:
        rsi_factor = max(-0.5, min(0.5, (30 - day.rsi) / 100))
        williams_factor = max(-0.5, min(0.5, (-80 - day.williams) / 100))
        adx_factor = max(-0.5, min(0.5, (day.adx - 25) / 100))
        return (rsi_factor * 0.3 + williams_factor * 0.4 + adx_factor * 0.3)

    def level3_dynamic_weights(self, day: TradingDay) -> float:
        adx_strength = day.adx / 100
        if adx_strength > 0.25:
            rsi_weight = 0.4
            williams_weight = 0.4
            adx_weight = 0.2
        else:
            rsi_weight = 0.3
            williams_weight = 0.3
            adx_weight = 0.4
        rsi_factor = max(-0.5, min(0.5, (30 - day.rsi) / 100))
        williams_factor = max(-0.5, min(0.5, (-80 - day.williams) / 100))
        adx_factor = max(-0.5, min(0.5, (day.adx - 25) / 100))
        return (rsi_factor * rsi_weight +
                williams_factor * williams_weight +
                adx_factor * adx_weight)

    # Level 4 methods
    def level4_conservative(self, score: float) -> int:
        if score >= 1:
            return 1
        if score <= -1:
            return -1
        return 0

    def level4_aggressive(self, score: float) -> int:
        if score > 0:
            return 1
        if score < 0:
            return -1
        return 0

    def level4_adaptive(self, score: float) -> int:
        if abs(score) < 0.2:
            return 0
        elif abs(score) < 0.4:
            return 1 if score > 0 else -1
        else:
            return 2 if score > 0 else -2

    def level4_trend_following(self, score: float) -> int:
        if score > 0.3:
            return 1
        elif score < -0.4:
            return -1
        return 0

    def evaluate_prediction_method(self, method_name: str, predictions: List[float], day: TradingDay,
                                   level1_method: Callable, level2_method: Callable,
                                   level3_method: Callable, level4_method: Callable) -> Tuple[int, float]:
        adjusted_values = level1_method(predictions)
        signal = level2_method(adjusted_values, day.close)
        tech_adjustment = level3_method(day)
        combined_score = signal + tech_adjustment
        final_decision = level4_method(combined_score)
        return final_decision, combined_score

    def calculate_prediction_error(self, predictions: List[float], actual_prices: List[float]) -> float:
        if len(predictions) > len(actual_prices):
            predictions = predictions[:len(actual_prices)]
        errors = [abs(pred - actual) for pred, actual in zip(predictions, actual_prices)]
        return sum(errors) / len(errors)

    def calculate_optimal_weight(self, pred1: List[float], pred2: List[float], actual: List[float]) -> WeightedPrediction:
        best_weight = 0.0
        min_error = float('inf')

        error1 = sum(abs(p - a) for p, a in zip(pred1, actual)) / len(actual)
        error2 = sum(abs(p - a) for p, a in zip(pred2, actual)) / len(actual)

        for w in range(11):
            weight = w / 10
            blended = [weight * p1 + (1 - weight) * p2 for p1, p2 in zip(pred1, pred2)]
            error = sum(abs(p - a) for p, a in zip(blended, actual)) / len(actual)
            if error < min_error:
                min_error = error
                best_weight = weight

        return WeightedPrediction(weight=best_weight, error=min_error, error1=error1, error2=error2)

    def rolling_window_backtest_group(self, group: List[TradingDay], symbol: str, group_index: int) -> List[Tuple]:
        """
        Implements rolling 10-day window backtesting for adaptive strategy selection.
        Returns list of (day_index, best_result, final_decision, trade_signals, final_position)
        """
        if len(group) < self.rolling_window_size + 1:
            print(f"Warning: Group too small for rolling windows ({len(group)} days)")
            return []
            
        rolling_results = []
        
        # Generate rolling windows for days 10-19 (0-indexed: days 9-18)
        for signal_day in range(self.rolling_window_size - 1, len(group) - 1):  # days 9-18
            # Create 10-day backtesting window ending at signal_day
            window_start = signal_day - self.rolling_window_size + 1  # 10 days back
            window_end = signal_day + 1
            backtest_window = group[window_start:window_end]
            
            print(f"Rolling window for signal day {signal_day}: using days {window_start}-{signal_day} ({len(backtest_window)} days)")
            
            # Find best strategy for this 10-day window
            window_results = self.evaluate_all_combinations_group(backtest_window, symbol, f"{group_index}_rolling_{signal_day}")
            
            if not window_results:
                continue
                
            # Find best strategy from this window
            best_result = max(window_results, key=lambda r: (r[0].profit / 100.0 * 100))
            
            # Unpack the result tuple
            if len(best_result) == 5:
                best_strategy, final_decision, trade_signals, final_position, l5_method = best_result
            else:
                best_strategy, final_decision, trade_signals, final_position = best_result
                l5_method = "L5_none"
            
            # Store this rolling window result
            rolling_results.append((
                signal_day,           # Which day this signal is for
                best_strategy,        # Best strategy found
                final_decision,       # Trading decision for this day
                trade_signals,        # Historical trade signals used to find best strategy
                final_position,       # Position after trade
                l5_method            # L5 method used
            ))
            
        return rolling_results

    def calculate_rolling_window_performance(self, rolling_results: List[Tuple], group: List[TradingDay]) -> Tuple[float, int, int, List]:
        """
        Calculate cumulative performance from rolling window trading signals.
        Returns (total_profit, total_trades, correct_trades, execution_log)
        """
        position = 0
        entry_price = 0.0
        total_profit = 0.0
        total_trades = 0
        correct_trades = 0
        execution_log = []
        
        shares_bought = 100.0 / group[0].open if group[0].open != 0 else 0.0
        
        for i, (signal_day, best_strategy, final_decision, trade_signals, final_position, l5_method) in enumerate(rolling_results):
            # Execute trade on the NEXT day's open (signal_day + 1)
            if signal_day + 1 >= len(group):
                break
                
            execution_day = group[signal_day + 1]
            
            if final_decision == 1 and position == 0:  # Buy signal
                execution_price = self.calculate_slippage_adjusted_price(execution_day.open, 1)
                position = 1
                entry_price = execution_price
                total_trades += 1
                execution_log.append(f"Day {signal_day + 1}: BUY at ${execution_price:.2f} (Strategy: {best_strategy.method})")
                
            elif final_decision == -1 and position == 1:  # Sell signal
                execution_price = self.calculate_slippage_adjusted_price(execution_day.open, -1)
                
                # ✅ APPLY TIME WEIGHTING HERE TOO
                base_trade_profit = (execution_price - entry_price) * shares_bought
                
                # Weight based on when the trade occurs in the rolling window period
                # signal_day ranges from 9-18 (0-indexed), so normalize to 0-1
                #time_weight = 0.5 + ((signal_day - 9) / 9)  # 0.5 to 1.5 weighting
                time_weight = 1
                weighted_trade_profit = base_trade_profit * time_weight
                
                total_profit += weighted_trade_profit
                position = 0
                
                if weighted_trade_profit > 0:
                    correct_trades += 1
                    
                execution_log.append(f"Day {signal_day + 1}: SELL at ${execution_price:.2f}, Profit: ${weighted_trade_profit:.2f} (weight: {time_weight:.2f})")
        
        # Close final position if holding
        if position == 1 and len(group) > 0:
            final_day = group[-1]
            exit_price = self.calculate_slippage_adjusted_price(final_day.close, -1)
            
            base_final_profit = (exit_price - entry_price) * shares_bought
            time_weight = 1.5  # Maximum weight for final position
            weighted_final_profit = base_final_profit * time_weight
            
            total_profit += weighted_final_profit
            
            if weighted_final_profit > 0:
                correct_trades += 1
                
            execution_log.append(f"Final: SELL at ${exit_price:.2f}, Profit: ${weighted_final_profit:.2f} (weight: {time_weight:.2f})")
        
        return total_profit, total_trades, correct_trades, execution_log
  
    def backtest_strategy_group(
        self,
        group: List[TradingDay],
        symbol: str,
        group_index: int,
        method_name: str,
        get_predictions: Callable,
        level1_method: Callable,
        level2_method: Callable,
        level3_method: Callable,
        level4_method: Callable,
        l5_precomputed: dict = None,
        api_key: str = "",
        api_secret: str = "",
        l5_method_name: str = "L5_none",
        news_cache: dict = None,
        time_weighting: bool = True  # ✅ Add time weighting parameter
    ) -> Tuple[PredictionResult, int, list, int]:
        position = 0
        entry_price = 0.0
        profit = 0.0
        total_trades = 0
        total_buys = 0
        correct_trades = 0
        prediction_errors = []
        n = len(group)
        final_decision = 0
        trade_signals = []
        buy_indices = []
        sell_indices = []
        holding = False
        
        total_slippage_cost = 0.0
        shares_bought = 100.0 / group[0].open if group[0].open != 0 else 0.0

        for i, day in enumerate(group[:-1]):
            predictions = get_predictions(day)
            decision, _ = self.evaluate_prediction_method(
                method_name, predictions, day,
                level1_method, level2_method,
                level3_method, level4_method
            )
            
            # Level 5 logic (unchanged)
            l5_value = l5_precomputed[day.date].get(l5_method_name, None) if l5_method_name != "L5_none" else None
            if l5_method_name == "L5_news_sentiment_v1":
                if l5_value is not None:
                    if l5_value > 0.2:
                        decision = 1
                    elif l5_value < -0.2:
                        decision = -1
            elif l5_method_name == "L5_news_sentiment_v2":
                if l5_value is not None:
                    if l5_value > 0.2:
                        decision = 1
                    elif l5_value < -0.2:
                        decision = -1
            elif l5_method_name == "L5_news_keyword_filter":
                if l5_value is not None:
                    if l5_value > 0:
                        decision = 1
                    elif l5_value < 0:
                        decision = -1
            elif l5_method_name == "L5_news_override":
                if l5_value is not None:
                    if l5_value > 0.5:
                        decision = 1
                    elif l5_value < -0.5:
                        decision = -1

            trade_signals.append((day.date, decision))
            final_decision = decision

            next_day = group[i + 1]
            if i + len(predictions) < n:
                future_prices = [d.close for d in group[i + 1:i + 1 + len(predictions)]]
                error = self.calculate_prediction_error(predictions, future_prices)
                prediction_errors.append(error)

            # ✅ TRADE EXECUTION WITH TIME WEIGHTING
            if decision == 1 and position == 0:  # Buy signal
                execution_price = self.calculate_slippage_adjusted_price(next_day.open, 1)
                position = 1
                entry_price = execution_price
                buy_indices.append(i+1)
                total_buys += 1
                total_trades += 1
                holding = True
                
                slippage_cost = execution_price - next_day.open
                total_slippage_cost += slippage_cost * shares_bought
                    
            elif decision == -1 and position == 1:  # Sell signal
                execution_price = self.calculate_slippage_adjusted_price(next_day.open, -1)
                position = 0
                exit_price = execution_price
                sell_indices.append(i+1)
                
                # ✅ APPLY TIME WEIGHTING TO TRADES
                base_trade_profit = (exit_price - entry_price) * shares_bought
                
                if time_weighting:
                    # Weight from 0.5 early to 1.5 late in the period
                    time_weight = 0.5 + (i / max(1, len(group) - 2))
                    weighted_trade_profit = base_trade_profit * time_weight
                else:
                    weighted_trade_profit = base_trade_profit
                    
                profit += weighted_trade_profit
                
                slippage_cost = next_day.open - execution_price
                total_slippage_cost += slippage_cost * shares_bought
                
                if weighted_trade_profit > 0:
                    correct_trades += 1
                total_trades += 1
                holding = False

        # ✅ HANDLE FINAL POSITION WITH TIME WEIGHTING
        if position == 1:
            final_day = group[-1]
            base_slippage = self.slippage_config['base_slippage_bps'] / 10000
            exit_price = final_day.close * (1 - base_slippage)
            
            base_trade_profit = (exit_price - entry_price) * shares_bought
            
            if time_weighting:
                time_weight = 1.5  # Maximum weight for final trades
                weighted_trade_profit = base_trade_profit * time_weight
            else:
                weighted_trade_profit = base_trade_profit
                
            profit += weighted_trade_profit
            sell_indices.append(len(group)-1)
            
            if weighted_trade_profit > 0:
                correct_trades += 1
            
            final_slippage_cost = final_day.close - exit_price
            total_slippage_cost += final_slippage_cost * shares_bought

        mean_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0

        def get_name(f):
            return getattr(f, '__name__', str(f)).replace('<lambda>', 'lambda_func')

        result = PredictionResult(
            method=method_name,
            level1_method=get_name(level1_method),
            level2_method=get_name(level2_method),
            level3_method=get_name(level3_method),
            level4_method=get_name(level4_method),
            profit=profit,
            total_trades=total_trades,
            total_buys=total_buys,
            correct_trades=correct_trades,
            mean_prediction_error=mean_prediction_error,
            level5_method=l5_method_name 
        )
        
        result.total_slippage_cost = total_slippage_cost
        return result, final_decision, trade_signals, position

    def evaluate_all_combinations_group(self, group: List[TradingDay], symbol: str, group_index: int) -> List[PredictionResult]:
        results = []
        combination_counter = 0  # Add this counter
        
        news_cache = {}
        for day in group[:-1]:
            if day.date not in news_cache:
                news_cache[day.date] = self.fetch_news(symbol, YOUR_KEY, YOUR_SECRET, day.date)



        l5_precomputed = {}
        for day in group[:-1]:
            news = news_cache[day.date]
            # v1: avg_sentiment
            sentiments = [TextBlob(article.get('headline', '')).sentiment.polarity for article in news if 'headline' in article]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            # v2/override: avg_score
            scores = []
            for article in news:
                h = TextBlob(article.get('headline', '')).sentiment.polarity
                c = TextBlob(article.get('content', '')).sentiment.polarity
                scores.append(0.4 * h + 0.6 * c)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            # keyword filter: score
            positive_keywords = ["beats", "record", "growth", "upgrade", "surge"]
            negative_keywords = ["misses", "downgrade", "lawsuit", "decline", "plunge"]
            keyword_score = 0
            for article in news:
                text = (article.get('headline', '') + " " + article.get('content', '')).lower()
                if any(word in text for word in positive_keywords):
                    keyword_score += 1
                if any(word in text for word in negative_keywords):
                    keyword_score -= 1
            l5_precomputed[day.date] = {
                "L5_news_sentiment_v1": avg_sentiment,
                "L5_news_sentiment_v2": avg_score,
                "L5_news_keyword_filter": keyword_score,
                "L5_news_override": avg_score,
                 "L5_none": None
            }

        prediction_methods = [
            ("pred_1_1", lambda day: day.pred_1_1),
            ("pred_1_2", lambda day: day.pred_1_2),
            ("pred_1_3", lambda day: day.pred_1_3),
            ("pred_2_1", lambda day: day.pred_2_1),
            ("pred_2_2", lambda day: day.pred_2_2),
            ("pred_2_3", lambda day: day.pred_2_3),
            ("pred_3_1", lambda day: day.pred_3_1),
            ("pred_3_2", lambda day: day.pred_3_2),
            ("pred_3_3", lambda day: day.pred_3_3),
            ("pred_4_1", lambda day: day.pred_4_1),
            ("pred_4_2", lambda day: day.pred_4_2),
            ("pred_4_3", lambda day: day.pred_4_3),
            ("pred_5_1", lambda day: day.pred_5_1),
            ("pred_5_2", lambda day: day.pred_5_2),
            ("pred_5_3", lambda day: day.pred_5_3),
            ("pred_6_1", lambda day: day.pred_6_1),
            ("pred_6_2", lambda day: day.pred_6_2),
            ("pred_6_3", lambda day: day.pred_6_3)
        ]

        base_level1_methods = [
            self.level1_raw,
            self.level1_simple_average,
            self.level1_weighted_average,
            self.level1_exponential_average,
            self.level1_median_filter
        ]
        level2_methods = [
            self.level2_simple_comparison,
            self.level2_trend_analysis,
            self.level2_threshold,
            self.level2_combined,
            self.level2_momentum,
            self.level2_volatility_based
        ]
        level3_methods = [
            self.level3_no_adjustment,
            self.level3_rsi,
            self.level3_williams,
            self.level3_combined,
            self.level3_adx,
            self.level3_all_combined,
            self.level3_dynamic_weights
        ]

        # Calculate total combinations
        l5_method_names = [
            "L5_news_sentiment_v1",
            "L5_news_sentiment_v2",
            "L5_news_keyword_filter",
            "L5_news_override",
            "L5_none"
        ]
        num_l5 = len(l5_method_names)
        
        num_pred = len(prediction_methods)
        num_l1 = len(base_level1_methods)
        num_l2 = len(level2_methods)
        num_l3 = len(level3_methods)
        num_l4 = 4  # Conservative, Aggressive, Adaptive, Trend-Following
        total_combinations = num_l5 * num_pred * num_l1 * num_l2 * num_l3 * num_l4

        best_strategy = None
        best_profit_pct = -float('inf')
        best_decision = 0
        best_final_position = 0

        for l5_method_name in l5_method_names:
            for method_name, get_predictions in prediction_methods:
                level4_methods = [
                    self.level4_conservative,
                    self.level4_aggressive,
                    self.level4_adaptive,
                    self.level4_trend_following
                ]

                for l1 in base_level1_methods:
                    for l2 in level2_methods:
                        for l3 in level3_methods:
                            for l4 in level4_methods:
                                combination_counter += 1
                                result, decision, trade_signals, final_position = self.backtest_strategy_group(
                                    group, symbol, group_index, method_name, get_predictions, l1, l2, l3, l4,
                                    api_key=YOUR_KEY, api_secret=YOUR_SECRET,
                                    l5_method_name=l5_method_name,
                                    news_cache=news_cache,
                                    l5_precomputed=l5_precomputed
                                    
                                )
                                results.append((result, decision, trade_signals, final_position, l5_method_name))
                                
                                # ✅ ADD THIS: Track best strategy
                                pct = (result.profit / 100.0 * 100)
                                if pct > best_profit_pct:
                                    best_profit_pct = pct
                                    best_strategy = result
                                    best_decision = decision
                                    best_final_position = final_position

        # Special L1 section - ADD THE SAME TRACKING HERE
        for special_l1 in [level1_confidence_blend, level1_error_adjusted]:
            for pred_num in range(1, 7):
                pred1_key = f'pred_{pred_num}_1'
                pred2_key = f'pred_{pred_num}_2'
                pred3_key = f'pred_{pred_num}_3'

                if not hasattr(group[0], pred1_key) or not hasattr(group[0], pred2_key):
                    continue

                for l5_method_name in l5_method_names:
                    for l2 in level2_methods:
                        for l3 in level3_methods:
                            for l4 in level4_methods:
                                # Around line 1350-1380, replace the get_special_preds function:

                                def get_special_preds(day, p1=pred1_key, p2=pred2_key):
                                    pred1 = getattr(day, p1)
                                    pred2 = getattr(day, p2)
                                    
                                    # ✅ FIX: Extract the base group_index from rolling window naming
                                    base_group_index = group_index
                                    if isinstance(group_index, str) and "_rolling_" in str(group_index):
                                        # Extract the original group index (e.g., "0_rolling_9" -> 0)
                                        base_group_index = int(str(group_index).split("_rolling_")[0])
                                    
                                    # ✅ FIX: Safely get optimal weights with proper error handling
                                    try:
                                        symbol_weights = self.optimal_weights.get(symbol, [{}])
                                        if isinstance(symbol_weights, list) and len(symbol_weights) > base_group_index:
                                            group_weights = symbol_weights[base_group_index]
                                            error_info = group_weights.get(pred_num, WeightedPrediction(0.5, 0.1))
                                        else:
                                            error_info = WeightedPrediction(0.5, 0.1)
                                    except (IndexError, TypeError, AttributeError):
                                        # Fallback if anything goes wrong
                                        error_info = WeightedPrediction(0.5, 0.1)

                                    if special_l1 == level1_confidence_blend:
                                        return special_l1(pred1, pred2, error_info.error1, error_info.error2)
                                    elif special_l1 == level1_error_adjusted:
                                        return special_l1(pred1, [error_info.error])


                                def passthrough(x): return x
                                passthrough.__name__ = special_l1.__name__

                                # FIX: Unpack 4 values
                                result, decision, trade_signals, final_position = self.backtest_strategy_group(
                                    group, symbol, group_index, pred3_key, get_special_preds, passthrough, l2, l3, l4,
                                    api_key=YOUR_KEY, api_secret=YOUR_SECRET,
                                    l5_method_name=l5_method_name,
                                    l5_precomputed=l5_precomputed,
                                    time_weighting=True,# <-- Pass L5 name
                                )
                                results.append((result, decision, trade_signals, final_position, l5_method_name))

                                # ✅ ADD THIS: Track best strategy here too
                                pct = (result.profit / 100.0 * 100)
                                if pct > best_profit_pct:
                                    best_profit_pct = pct
                                    best_strategy = result
                                    best_decision = decision
                                    best_final_position = final_position

        # Now fix the final decision logic:
        if not hasattr(self, 'final_decisions'):
            self.final_decisions = {}
        if not hasattr(self, 'final_positions'):
            self.final_positions = {}

        # Use the actual best strategy's final decision and position
        self.final_decisions[symbol] = "Buy" if best_decision == 1 else ("Sell" if best_decision == -1 else "Hold")
        self.final_positions[symbol] = best_final_position

        return results



    def _calculate_buy_and_hold_profit(self, group: List[TradingDay]) -> Tuple[float, float]:
        if len(group) < 2:
            return 0.0, 0.0
        entry_price = group[0].open
        exit_price = group[-1].close
        shares_bought = 100.0 / entry_price if entry_price != 0 else 0.0
        profit = (exit_price - entry_price) * shares_bought
        profit_percentage = (profit / 100.0 * 100) if entry_price != 0 else 0
        return profit, profit_percentage

    def _calculate_perfect_profit(self, group: List[TradingDay]) -> Tuple[float, float, int, List[Tuple[str, str, float]]]:
        """
        Computes the maximum possible profit using next-day open prices.
        It sums every positive difference between consecutive open prices.
        """
        shares_bought = 100.0 / group[0].open if group[0].open != 0 else 0.0
        max_profit = 0.0
        trades = []
        n = len(group)
        for i in range(n - 1):
            diff = group[i + 1].open - group[i].open
            if diff > 0:
                max_profit += diff * shares_bought
                trades.append((group[i + 1].date, 'TRADE', diff * shares_bought))
        initial_investment = 100.0
        profit_pct = (max_profit / initial_investment * 100) if initial_investment != 0 else 0
        return max_profit, profit_pct, len(trades), trades

    def run_all_backtests(self, json_path):
        self.results_by_stock = {}
        self.cumulative_results_by_stock = {}
        global_initial = 0.0
        global_profit = 0.0

        # ✅ MODIFY TO SUPPORT ROLLING WINDOWS
        stock_args = [(symbol, groups, json_path, self.use_rolling_windows) for symbol, groups in self.trading_data.items()]

        with multiprocessing.Pool() as pool:
            results = pool.map(process_stock_rolling, stock_args)
            
        # Collect results
        self.final_decisions = {}
        self.final_positions = {}
        
        for symbol, group_results, total_initial_buy_hold, total_profit_buy_hold, cumulative, groups, final_decisions, final_positions in results:
            # Merge final_decisions and final_positions from each worker
            self.final_decisions.update(final_decisions)
            self.final_positions.update(final_positions)
            
            # ... rest of your existing collection logic remains the same ...
            self.results_by_stock[symbol] = []
            
            # ✅ FIX LINE 1469 - Complete the for loop:
            for group_index, results_with_plots, group, rolling_results in group_results:  # Add rolling_results here
                results = [r[0] for r in results_with_plots]

                best_result = max(
                    results_with_plots,
                    key=lambda r: (r[0].profit / 100.0 * 100)
                )

                # Unpack all five values (including L5 method                    
                if len(best_result) == 5:
                    best_strategy, _, trade_signals, _, best_l5_method = best_result
                else:
                    best_strategy, _, trade_signals, _ = best_result
                    best_l5_method = "L5_none"

                # ✅ ADD THE ROLLING WINDOW PLOT LOGIC HERE:
                if self.use_rolling_windows and rolling_results:
                    best_plot_html = generate_rolling_window_plot(group, rolling_results, symbol, group_index)
                else:
                    best_plot_html = generate_inline_plot(group, trade_signals, symbol, group_index)

                # ✅ Ensure group-best strategy is counted in cumulative totals
                best_key = (best_strategy.method, best_strategy.level1_method, best_strategy.level2_method,
                    best_strategy.level3_method, best_strategy.level4_method, best_strategy.level5_method)
                
                if best_key not in cumulative:
                    cumulative[best_key] = {
                        'total_profit': 0.0,
                        'total_initial': 0.0,
                        'total_trades': 0,
                        'total_buys': 0,
                        'total_correct': 0
                    }
                cumulative[best_key]['total_profit'] += best_strategy.profit
                cumulative[best_key]['total_initial'] += 100.0
                cumulative[best_key]['total_trades'] += best_strategy.total_trades
                cumulative[best_key]['total_buys'] += best_strategy.total_buys
                cumulative[best_key]['total_correct'] += best_strategy.correct_trades

                perfect_result = self._calculate_perfect_profit(group)
                open_prices = [d.open for d in group]
                trend_profit, trend_trades, trend_winrate = trend_following_strategy(open_prices)
                bh_profit, bh_pct = self._calculate_buy_and_hold_profit(group)
                
                self.results_by_stock[symbol].append((
                    group_index, best_strategy, perfect_result, trend_profit, trend_trades, trend_winrate,
                    [r[0] for r in results_with_plots], group[0].open, bh_profit, bh_pct, best_plot_html
                ))

                if len(groups) == 1:
                    self.cumulative_results_by_stock[symbol] = {
                        best_key: {
                            'total_profit': best_strategy.profit,
                            'total_initial': 100.0,
                            'total_trades': best_strategy.total_trades,
                            'total_buys': best_strategy.total_buys,
                            'total_correct': best_strategy.correct_trades
                        }
                    }

                for res in results:
                    key = (res.method, res.level1_method, res.level2_method, res.level3_method, res.level4_method, res.level5_method)
                    if key not in cumulative:
                        cumulative[key] = {
                            'total_profit': 0.0,
                            'total_initial': 0.0,
                            'total_trades': 0,
                            'total_buys': 0,
                            'total_correct': 0
                        }
                    cumulative[key]['total_profit'] += res.profit
                    cumulative[key]['total_initial'] += 100.0
                    cumulative[key]['total_trades'] += res.total_trades
                    cumulative[key]['total_buys'] += res.total_buys
                    cumulative[key]['total_correct'] += res.correct_trades
                
            self.buy_hold_by_stock[symbol] = (
                total_profit_buy_hold,
                (total_profit_buy_hold / total_initial_buy_hold * 100) if total_initial_buy_hold != 0 else 0
            )
            global_initial += total_initial_buy_hold
            global_profit += total_profit_buy_hold
            print(f"Completed processing stock: {symbol}")

        self.global_buy_hold = (
            global_profit,
            (global_profit / global_initial * 100) if global_initial != 0 else 0
        )

        # Determine best cumulative strategy per stock
        self.best_strategy_by_stock = {}
        self.used_stocks = []
        self.excluded_stocks = []
        self.portfolio_stocks = []
        stock_performance = []

        for symbol, cum in self.cumulative_results_by_stock.items():
            best_key = None
            best_pct = -float('inf')
            best_stats = None
            for key, stats in cum.items():
                initial = stats['total_initial']
                profit_pct = (stats['total_profit'] / initial * 100) if initial != 0 else 0
                if profit_pct > best_pct:
                    best_pct = profit_pct
                    best_key = key
                    best_stats = stats

            bh_profit, bh_pct = self.buy_hold_by_stock.get(symbol, (0, 0))
            if best_pct > bh_pct:
                self.best_strategy_by_stock[symbol] = (best_key, best_pct, best_stats)
                self.used_stocks.append(symbol)
                stock_performance.append((symbol, best_pct))
            else:
                self.excluded_stocks.append(symbol)

        stock_performance.sort(key=lambda x: x[1], reverse=True)
        self.portfolio_stocks = [s[0] for s in stock_performance[:5]]

        global_cumulative = {}
        for symbol in self.portfolio_stocks:
            cum = self.cumulative_results_by_stock.get(symbol, {})
            for key, stats in cum.items():
                if key not in global_cumulative:
                    global_cumulative[key] = {
                        'total_profit': 0.0,
                        'total_initial': 0.0,
                        'total_trades': 0,
                        'total_buys': 0,
                        'total_correct': 0
                                       }
                global_cumulative[key]['total_profit'] += stats['total_profit']
                global_cumulative[key]['total_initial'] += stats['total_initial']
                global_cumulative[key]['total_trades'] += stats['total_trades']
                global_cumulative[key]['total_buys'] += stats['total_buys']
                global_cumulative[key]['total_correct'] += stats['total_correct']

        best_global_key = None
        best_global_pct = -float('inf')
        for key, stats in global_cumulative.items():
            initial = stats['total_initial']
            profit_pct = (stats['total_profit'] / initial * 100) if initial != 0 else 0
            if profit_pct > best_global_pct:
                best_global_pct = profit_pct
                best_global_key = key

        if best_global_key:
            self.global_best = (
                best_global_key,
                best_global_pct,
                global_cumulative[best_global_key]
            )
        else:
            self.global_best = None

        self.per_stock_best_portfolio = {
            'total_initial': 0.0,
            'total_profit': 0.0,
            'total_trades': 0,
            'total_buys': 0,
            'total_correct': 0
        }

        self.global_unified_strategy = {
            'strategy_key': None,
            'total_initial': 0.0,
            'total_profit': 0.0,
            'total_trades': 0,
            'total_buys': 0,
            'total_correct': 0
        }

        top_stocks = [s[0] for s in stock_performance[:5]]
        strategy_sums = {}
        for symbol in top_stocks:
            for key, stats in self.cumulative_results_by_stock[symbol].items():
                if key not in strategy_sums:
                    strategy_sums[key] = {
                        'total_initial': 0.0,
                        'total_profit': 0.0,
                        'total_trades': 0,
                        'total_buys': 0,
                        'total_correct': 0
                    }
                strategy_sums[key]['total_initial'] += stats['total_initial']
                strategy_sums[key]['total_profit'] += stats['total_profit']
                strategy_sums[key]['total_trades'] += stats['total_trades']
                strategy_sums[key]['total_buys'] += stats['total_buys']
                strategy_sums[key]['total_correct'] += stats['total_correct']

        best_key = None
        best_pct = -float('inf')
        for key, stats in strategy_sums.items():
            initial = stats['total_initial']
            if initial > 0:
                pct = stats['total_profit'] / initial * 100
                if pct > best_pct:
                    best_key = key
                    best_pct = pct
                    self.global_unified_strategy = {
                        'strategy_key': key,
                        'total_initial': initial,
                        'total_profit': stats['total_profit'],
                        'total_trades': stats['total_trades'],
                        'total_buys': stats['total_buys'],
                        'total_correct': stats['total_correct'],
                        'profit_pct': pct,
                        'accuracy': (stats['total_correct'] / stats['total_buys'] * 100)
                        if stats['total_buys'] > 0 else 0.0
                    }

        for symbol in self.portfolio_stocks:
            best_key, _, stats = self.best_strategy_by_stock[symbol]
            self.per_stock_best_portfolio['total_initial'] += stats['total_initial']
            self.per_stock_best_portfolio['total_profit'] += stats['total_profit']
            self.per_stock_best_portfolio['total_trades'] += stats['total_trades']
            self.per_stock_best_portfolio['total_correct'] += stats['total_correct']
            self.per_stock_best_portfolio['total_buys'] += stats['total_buys']

        if self.per_stock_best_portfolio['total_initial'] > 0:
            profit_pct = (
                self.per_stock_best_portfolio['total_profit'] /
                self.per_stock_best_portfolio['total_initial'] * 100
            )
            accuracy = (
                self.per_stock_best_portfolio['total_correct'] /
                self.per_stock_best_portfolio['total_buys'] * 100
                if self.per_stock_best_portfolio['total_buys'] > 0 else 0
            )
            self.per_stock_best_portfolio['profit_pct'] = profit_pct
            self.per_stock_best_portfolio['accuracy'] = accuracy
        else:
            self.per_stock_best_portfolio['profit_pct'] = 0.0
            self.per_stock_best_portfolio['accuracy'] = 0.0

    def export_results_to_html(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'strategy_results_{timestamp}.html'
        if not filename.endswith('.html'):
            filename = f'{filename}.html'

        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)

        css_styles = get_sleek_html_css()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Trading Strategy Results</title>
            {css_styles}
            <style>
            .search-box {{
                margin: 10px 0;
                padding: 5px;
                width: 250px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th {{
                background-color: #f2f2f2;
                cursor: pointer;
                padding: 10px;
            }}
            td {{
                padding: 8px;
                text-align: center;
            }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stock-section {{
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 12px;
                background-color: #fff;
                margin-bottom: 40px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            }}
            </style>
        </head>
        <body>
        <div class="container">
            <h1>📈 Trading Strategy Results</h1>
        """

        # ... existing per-stock/group HTML generation code remains unchanged ...

        for symbol, group_results in self.results_by_stock.items():
            bh_profit_stock, bh_pct_stock = self.buy_hold_by_stock.get(symbol, (0, 0))
            html_content += f"<div class='stock-section'><h2>Stock: {symbol}</h2>"

            for (group_index, best_strategy, perfect_result, trend_profit, trend_trades, trend_winrate,
                results, group_initial, group_bh_profit, group_bh_pct, plot_html) in group_results:
            
                best_pct = (best_strategy.profit / 100.0 * 100) if best_strategy.profit is not None else 0
                perfect_profit, perfect_pct, perfect_trades, _ = perfect_result

                pred_num = int(best_strategy.method.split('_')[1])
                group_weights = self.optimal_weights.get(symbol, [{}])[group_index].get(pred_num)

                # Extra info for L1
                extra_l1 = ""
                if best_strategy.level1_method == "level1_confidence_blend" and group_weights:
                    err1 = getattr(group_weights, 'error1', None)
                    err2 = getattr(group_weights, 'error2', None)
                    if err1 is not None and err2 is not None:
                        extra_l1 = f" <span style='color:gray'>(Errors: pred1={err1:.4f}, pred2={err2:.4f})</span>"
                elif best_strategy.level1_method == "level1_error_adjusted" and group_weights:
                    err = getattr(group_weights, 'error', None)
                    if err is not None:
                        extra_l1 = f" <span style='color:gray'>(Bias Error: {err:.4f})</span>"

                html_content += f"""
                <table>
                    <tr>
                        <th>Best Strategy</th>
                        <th>Profit ($)</th>
                        <th>% Profit</th>
                        <th>Total Trades</th>
                        <th>Accuracy (%)</th> <!-- Added column header -->
                    </tr>
                    <tr class="comparison-row">
                        <td>
                            <span class="strategy-label">Prediction:</span> {best_strategy.method}<br>
                            <span class="strategy-label">L1:</span> {best_strategy.level1_method}{extra_l1}<br>
                            <span class="strategy-label">L2:</span> {best_strategy.level2_method}<br> 
                            <span class="strategy-label">L3:</span> {best_strategy.level3_method}<br>
                            <span class="strategy-label">L4:</span> {best_strategy.level4_method}<br>
                            <span class="strategy-label">L5:</span> {best_strategy.level5_method}
                        </td>
                        <td>${best_strategy.profit:.2f}</td>
                        <td>{best_pct:.2f}%</td>
                        <td>{best_strategy.total_trades}</td>
                        <td>{(best_strategy.correct_trades / best_strategy.total_buys * 100) if best_strategy.total_buys > 0 else 0:.2f}%</td>
                    </tr>
                </table>
                <p><strong>📊 Buy and Hold:</strong> Profit: ${group_bh_profit:.2f}, % Profit: {group_bh_pct:.2f}%</p>
                <p><strong>🎯 Perfect Trading:</strong> Profit: ${perfect_profit:.2f}, % Profit: {perfect_pct:.2f}%, Trades: {perfect_trades}</p>
                <p><strong>📉 Trend-Following:</strong> Profit: ${trend_profit:.2f}, Trades: {trend_trades}, Win Rate: {trend_winrate * 100:.2f}%</p>
                {plot_html}
                """

                if self.table:
                    html_content += generate_sortable_html_table(group_index, symbol, results, group_initial)

            # Cumulative Summary
            best_key = None
            best_stats = None
            best_pct = -float('inf')

            for key, stats in self.cumulative_results_by_stock.get(symbol, {}).items():
                if stats['total_initial'] > 0:
                    pct = stats['total_profit'] / stats['total_initial'] * 100
                    if pct > best_pct:
                        best_pct = pct
                        best_key = key
                        best_stats = stats

            if best_key:
                method, l1, l2, l3, l4, l5 = best_key
                pred_num = int(method.split('_')[1])
                group_weights = self.optimal_weights.get(symbol, [{}])[0].get(pred_num)

                extra_l1_cumulative = ""
                if l1 == "level1_confidence_blend" and group_weights:
                    err1 = getattr(group_weights, 'error1', None)
                    err2 = getattr(group_weights, 'error2', None)
                    if err1 is not None and err2 is not None:
                        extra_l1_cumulative = f" <span style='color:gray'>(Errors: pred1={err1:.4f}, pred2={err2:.4f})</span>"
                elif l1 == "level1_error_adjusted" and group_weights:
                    err = getattr(group_weights, 'error', None)
                    if err is not None:
                        extra_l1_cumulative = f" <span style='color:gray'>(Bias Error: {err:.4f})</span>"

                if len(self.trading_data.get(symbol, [])) > 1:

                    html_content += f"""
                        <h3>📌 Cumulative for {symbol}</h3>
                        <p><strong>Best Strategy:</strong> Prediction {method}, L1: {l1}{extra_l1_cumulative}, L2: {l2}, L3: {l3}, L4: {l4}, L5: {l5}</p>
                        <p><strong>Cumulative Profit %:</strong> {best_pct:.2f}%</p>
                        """
            
            else:
                html_content += f"""
                <h3>📌 Cumulative for {symbol}</h3>
                <p>No valid strategy results found.</p>
                """
            html_content += "</div>"

        # Portfolio + Unified Summary
        if hasattr(self, 'used_stocks') and hasattr(self, 'excluded_stocks'):
            html_content += """
            <div class='stock-section'>
            <h2>📌 Stock Inclusion Summary</h2>
            <p><strong>✅ Strategy Beat Buy & Hold:</strong> {}</p>
            <p><strong>🚫 Strategy Underperformed (excluded):</strong> {}</p>
            <p><strong>💼 Top {} Stocks Used in Portfolio:</strong></p>
            <ul>
            """.format(', '.join(self.used_stocks), ', '.join(self.excluded_stocks), len(self.portfolio_stocks))

            for symbol in self.portfolio_stocks:
                best_key, _, stats = self.best_strategy_by_stock.get(symbol, (None, 0, {}))
                if best_key:
                    method, l1, l2, l3, l4, l5 = best_key
                    pred_num = int(method.split('_')[1])
                    group_weights = self.optimal_weights.get(symbol, [{}])[0].get(pred_num)
                    extra_l1 = ""
                    if l1 == "level1_confidence_blend" and group_weights:
                        err1 = getattr(group_weights, 'error1', None)
                        err2 = getattr(group_weights, 'error2', None)
                        if err1 is not None and err2 is not None:
                            extra_l1 = f" <span style='color:gray'>(Errors: pred1={err1:.4f}, pred2={err2:.4f})</span>"
                    elif l1 == "level1_error_adjusted" and group_weights:
                        err = getattr(group_weights, 'error', None)
                        if err is not None:
                            extra_l1 = f" <span style='color:gray'>(Bias Error: {err:.4f})</span>"
                    l5_method = "L5_none"
                    for group_result in self.results_by_stock[symbol]:
                        _, best_strategy, *_ = group_result
                        if (best_strategy.method, best_strategy.level1_method, best_strategy.level2_method,
                            best_strategy.level3_method, best_strategy.level4_method) == best_key:
                            l5_method = getattr(best_strategy, "level5_method", "L5_none")
                            break

                    html_content += f"<li><strong>{symbol}</strong>: Prediction {method}, L1: {l1}{extra_l1}, L2: {l2}, L3: {l3}, L4: {l4}, L5: {l5_method}</li>"

            html_content += "</ul>"

            portfolio_initial = self.per_stock_best_portfolio.get('total_initial', 0)
            portfolio_profit = self.per_stock_best_portfolio.get('total_profit', 0)
            portfolio_trades = self.per_stock_best_portfolio.get('total_trades', 0)
            portfolio_buys = self.per_stock_best_portfolio.get('total_buys', 0)
            portfolio_correct = self.per_stock_best_portfolio.get('total_correct', 0)

            portfolio_profit_pct = (portfolio_profit / portfolio_initial * 100) if portfolio_initial != 0 else 0
            portfolio_accuracy = (portfolio_correct / portfolio_buys * 100) if portfolio_buys != 0 else 0

            # ✅ FIX: All-Stocks Metrics (ALL stocks, including losers)
            all_initial = 0.0
            all_profit = 0.0
            all_trades = 0
            all_buys = 0
            all_correct = 0

            all_symbols = list(self.trading_data.keys())
            print("all_symbols")
            print(all_symbols)
            for symbol in all_symbols:
                # Get the best result for each stock (even if it lost to buy-and-hold)
                if symbol in self.results_by_stock:
                    for group_result in self.results_by_stock[symbol]:
                        _, best_strategy, *_ = group_result
                        all_initial += 100.0  # Each group gets $100
                        all_profit += best_strategy.profit
                        all_trades += best_strategy.total_trades
                        all_buys += best_strategy.total_buys
                        all_correct += best_strategy.correct_trades

            all_profit_pct = (all_profit / all_initial * 100) if all_initial != 0 else 0
            all_accuracy = (all_correct / all_buys * 100) if all_buys != 0 else 0

            # Buy & Hold calculations remain the same...
            portfolio_bh_profit = sum(self.buy_hold_by_stock[s][0] for s in self.portfolio_stocks)
            portfolio_bh_initial = 100.0 * len(self.portfolio_stocks)
            portfolio_bh_pct = (portfolio_bh_profit / portfolio_bh_initial * 100) if portfolio_bh_initial != 0 else 0

            all_bh_profit = sum(self.buy_hold_by_stock[s][0] for s in all_symbols)
            all_bh_initial = 100.0 * len(all_symbols)
            all_bh_pct = (all_bh_profit / all_bh_initial * 100) if all_bh_initial != 0 else 0
   
            # Update the HTML output
            html_content += f"""
            <h3>📊 Portfolio Metrics (Top {len(self.portfolio_stocks)} Stocks)</h3>
            <p><strong>📈 Strategy:</strong> Profit: ${portfolio_profit:.2f}, % Profit: {portfolio_profit_pct:.2f}%, Accuracy: {portfolio_accuracy:.2f}%</p>
            <p><strong>📊 Buy & Hold:</strong> Profit: ${portfolio_bh_profit:.2f}, % Profit: {portfolio_bh_pct:.2f}%</p>
            
            <hr>
            <h3>📊 All-Stocks Metrics ({len(all_symbols)} Total Stocks)</h3>
            <p><strong>📈 Strategy:</strong> Profit: ${all_profit:.2f}, % Profit: {all_profit_pct:.2f}%, Accuracy: {all_accuracy:.2f}%</p>
            <p><strong>📊 Buy & Hold:</strong> Profit: ${all_bh_profit:.2f}, % Profit: {all_bh_pct:.2f}%</p>
            </div>
            """

        # Final Strategy-Based Signal section (always show if available)
        if hasattr(self, 'final_decisions') and self.final_decisions:
            html_content += """
            <div class='stock-section'>
            <h4>📍 Final Strategy-Based Signal</h4>
            <table>
                <tr><th>Stock</th><th>Signal</th><th>Action Needed</th></tr>
            """
            for stock in sorted(self.final_decisions.keys()):
                signal = self.final_decisions[stock]
                holding = self.final_positions.get(stock, 0)  # 1 if holding, 0 if not
                # Action logic:
                if signal == "Buy" and not holding:
                    action = "Buy"
                elif signal == "Sell" and holding:
                    action = "Sell"
                else:
                    action = "Hold / No Action"
                if signal not in ("Buy", "Sell"):
                    signal = "Hold / No Action"
                html_content += f"<tr><td>{stock}</td><td>{signal}</td><td>{action}</td></tr>"
            html_content += """
            </table>
            </div>
            """

        html_content += """
        </div></body></html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Results exported to: {filepath}")

    def get_news_time_window_for_day(self, day_date: str) -> Tuple[str, str]:
        dt = datetime.strptime(day_date, "%Y-%m-%d").replace(hour=21, minute=0, second=0)
        hours = 72 if dt.weekday() == 0 else 24
        start = (dt - timedelta(hours=hours)).isoformat() + "Z"
        end = dt.isoformat() + "Z"
        return start, end

    def fetch_news(self, symbol: str, api_key: str, api_secret: str, day_date: str) -> List[Dict]:
        """
        Fetches news for a symbol using Alpaca API for the window ending at 9pm UTC of day_date.
        Returns a list of news article dicts.
        """
        start, end = self.get_news_time_window_for_day(day_date)
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": symbol,
            "start": start,
            "end": end,
            "limit": 50,
            "include_content": "true",
            "exclude_contentless": "true",
            "sort": "desc"
        }
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        response = requests.get(url, headers=headers, params=params)
        try:
            return response.json().get("news", [])
        except Exception:
            return []
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, '..', 'json', 'trading_database_export.json')
    json_path = os.path.abspath(json_path)

    tester = StrategyTester(json_path)
    
    # ✅ CONFIGURE ROLLING WINDOWS
    tester.use_rolling_windows = True  # Set to False for original behavior
    tester.rolling_window_size = 10    # 10-day windows
    tester.table = False               # Don't show detailed tables for rolling windows
    
    # ✅ VERIFY CONFIGURATION
    print(f"Using rolling windows: {tester.use_rolling_windows}")
    print(f"Rolling window size: {tester.rolling_window_size}")
    print(f"Show detailed tables: {tester.table}")
    
    tester.run_all_backtests(json_path)
    tester.export_results_to_html("rolling_window_results")
