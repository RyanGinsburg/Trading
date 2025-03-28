from typing import Callable, List, Tuple
import math

# ✅ Confidence-based Level 4 strategies

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
