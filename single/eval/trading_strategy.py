import json
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import numpy as np # type: ignore
import csv
from datetime import datetime
import os

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
    correct_trades: int
    mean_prediction_error: float
    
    def __str__(self):
        accuracy = (self.correct_trades/self.total_trades*100) if self.total_trades > 0 else 0
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
    weight: float  # Weight for prediction 1 (1 - weight will be used for prediction 2)
    error: float   # Mean absolute error of the weighted prediction

class StrategyTester:
    def __init__(self, json_file: str):
        self.trading_data = self._load_data(json_file)
        self.results: List[PredictionResult] = []

    def _load_data(self, json_file: str) -> Dict[str, List[TradingDay]]:
        with open(json_file, 'r') as f:
            raw_data = json.load(f)

        processed_data = {}
        optimal_weights = {}  # Store optimal weights for each prediction pair
        
        for symbol, data in raw_data.items():
            trading_days = []
            
            # First pass to load basic data and calculate optimal weights
            for i, entry in enumerate(data[0]['data']):
                # Convert prediction strings to float lists
                predictions = {}
                for key in entry.keys():
                    if key.startswith('pred_'):
                        predictions[key] = [float(x) for x in entry[key].split(',')]
            
                # If we have enough future data, calculate optimal weights
                if i < len(data[0]['data']) - max(len(predictions['pred_1_1']), 1):
                    future_prices = [float(data[0]['data'][i + j + 1]['close']) 
                                   for j in range(len(predictions['pred_1_1']))]
                    
                    # Calculate optimal weights for each prediction pair
                    for pred_num in range(1, 7):
                        pred1_key = f'pred_{pred_num}_1'
                        pred2_key = f'pred_{pred_num}_2'
                        weight_info = self.calculate_optimal_weight(
                            predictions[pred1_key],
                            predictions[pred2_key],
                            future_prices
                        )
                        if pred_num not in optimal_weights:
                            optimal_weights[pred_num] = weight_info

            # Second pass to create TradingDay objects with weighted predictions
            for entry in data[0]['data']:
                predictions = {}
                for key in entry.keys():
                    if key.startswith('pred_'):
                        predictions[key] = [float(x) for x in entry[key].split(',')]
                
                # Calculate weighted predictions
                for pred_num in range(1, 7):
                    weight = optimal_weights.get(pred_num, WeightedPrediction(0.5, 0.0)).weight
                    pred1_key = f'pred_{pred_num}_1'
                    pred2_key = f'pred_{pred_num}_2'
                    pred3_key = f'pred_{pred_num}_3'
                    
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
            
            processed_data[symbol] = trading_days
            
            # Store weights information for HTML display
            self.optimal_weights = {
                f"pred_{num}": weight_info
                for num, weight_info in optimal_weights.items()
            }
        
        return processed_data

    def export_results_to_html(self, filename: str = None):
        """Export results to a visually appealing HTML file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'strategy_results_{timestamp}.html'
        
        if not filename.endswith('.html'):
            filename = f'{filename}.html'
        
        # Get the directory where trading_strategy.py is (eval/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Create results directory in the eval directory
        results_dir = os.path.join(current_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create full filepath
        filepath = os.path.join(results_dir, filename)
        
        perfect_profit, perfect_percentage, perfect_trades, perfect_trade_details = self.calculate_perfect_profit()
        buy_hold_profit, buy_hold_percentage = self.calculate_buy_and_hold_profit()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Results</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1400px; margin: auto; }}
                h1, h2 {{ color: #2c3e50; }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .perfect-trades {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 10px;
                    margin: 20px 0;
                }}
                .trade-card {{
                    background-color: #fff;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .buy {{ color: #27ae60; }}
                .sell {{ color: #e74c3c; }}
                .controls {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .controls select, .controls input {{
                    padding: 5px;
                    margin-right: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{ 
                    background-color: #2c3e50; 
                    color: white;
                    cursor: pointer;
                }}
                th:hover {{
                    background-color: #34495e;
                }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .profit {{ color: #27ae60; }}
                .loss {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading Strategy Results</h1>
                
                <div class="summary-box">
                    <h2>Benchmark Comparisons</h2>
                    <p><strong>Perfect Trading:</strong> <span class="profit">${perfect_profit:.2f} ({perfect_percentage:.1f}%)</span> ({perfect_trades} trades)</p>
                    <p><strong>Buy and Hold:</strong> <span class="profit">${buy_hold_profit:.2f} ({buy_hold_percentage:.1f}%)</span> (1 trade)</p>
                </div>

                <div class="summary-box">
                    <h2>Optimal Prediction Weights</h2>
                    <table>
                        <tr>
                            <th>Prediction Pair</th>
                            <th>Weight (Pred 1)</th>
                            <th>Weight (Pred 2)</th>
                            <th>Mean Error</th>
                        </tr>
"""

        for pred_num in range(1, 7): #type: ignore
            weight_info = self.optimal_weights.get(f"pred_{pred_num}")
            if weight_info:
                html_content += f"""
                        <tr>
                            <td>Prediction {pred_num}</td>
                            <td>{weight_info.weight:.2f}</td>
                            <td>{1 - weight_info.weight:.2f}</td>
                            <td>${weight_info.error:.2f}</td>
                        </tr>
"""

        html_content += """
                    </table>
                </div>

                <div class="summary-box">
                    <h2>Perfect Trading Details</h2>
                    <div class="perfect-trades">
"""

        # Add perfect trade details
        for date, action, price in perfect_trade_details:
            color_class = "buy" if action == "BUY" else "sell"
            html_content += f"""
                        <div class="trade-card">
                            <p>{date}</p>
                            <p class="{color_class}"><strong>{action}</strong> at ${price:.2f}</p>
                        </div>
"""

        html_content += """
                    </div>
                </div>

                <div class="controls">
                    <h2>Filter and Sort Options</h2>
                    <input type="text" id="searchInput" placeholder="Search strategies...">
                    <select id="profitFilter">
                        <option value="all">All Results</option>
                        <option value="profit">Profitable Only</option>
                        <option value="loss">Loss Only</option>
                    </select>
                    <label>Minimum Accuracy: <input type="number" id="minAccuracy" min="0" max="100" value="0">%</label>
                </div>

                <table id="strategyTable">
                    <thead>
                        <tr>
                            <th data-sort="rank">Rank</th>
                            <th data-sort="method">Method</th>
                            <th data-sort="level1">Level 1</th>
                            <th data-sort="level2">Level 2</th>
                            <th data-sort="level3">Level 3</th>
                            <th data-sort="level4">Level 4</th>
                            <th data-sort="profit">Profit</th>
                            <th data-sort="return">Return %</th>
                            <th data-sort="perfPercent">% of Perfect</th>
                            <th data-sort="trades">Trades</th>
                            <th data-sort="accuracy">Accuracy</th>
                            <th data-sort="error">Pred. Error</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add all strategies
        initial_investment = next(iter(self.trading_data.values()))[0].open
        for i, result in enumerate(self.results, 1):
            accuracy = (result.correct_trades/result.total_trades*100) if result.total_trades > 0 else 0
            profit_percentage = (result.profit/perfect_profit*100) if perfect_profit > 0 else 0
            strategy_percentage = self.calculate_profit_percentage(result.profit, initial_investment)
            profit_class = "profit" if result.profit > 0 else "loss"
            
            html_content += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{result.method}</td>
                            <td>{result.level1_method}</td>
                            <td>{result.level2_method}</td>
                            <td>{result.level3_method}</td>
                            <td>{result.level4_method}</td>
                            <td class="{profit_class}" data-value="{result.profit}">${result.profit:.2f}</td>
                            <td data-value="{strategy_percentage}">{strategy_percentage:.1f}%</td>
                            <td data-value="{profit_percentage}">{profit_percentage:.1f}%</td>
                            <td>{result.total_trades}</td>
                            <td data-value="{accuracy}">{accuracy:.1f}%</td>
                            <td data-value="{result.mean_prediction_error}">${result.mean_prediction_error:.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
            <script>
                $(document).ready(function() {
                    // Sorting functionality
                    $('th').click(function() {
                        var table = $(this).parents('table').eq(0);
                        var rows = table.find('tr:gt(0)').toArray().sort(comparer($(this).index()));
                        this.asc = !this.asc;
                        if (!this.asc) rows = rows.reverse();
                        for (var i = 0; i < rows.length; i++) {
                            table.append(rows[i]);
                        }
                    });

                    function comparer(index) {
                        return function(a, b) {
                            var valA = getCellValue(a, index);
                            var valB = getCellValue(b, index);
                            return $.isNumeric(valA) && $.isNumeric(valB) ?
                                valA - valB : valA.toString().localeCompare(valB);
                        }
                    }

                    function getCellValue(row, index) {
                        var cell = $(row).children('td').eq(index);
                        return cell.data('value') || cell.text();
                    }

                    // Filtering functionality
                    $('#searchInput').on('keyup', filterTable);
                    $('#profitFilter').on('change', filterTable);
                    $('#minAccuracy').on('input', filterTable);

                    function filterTable() {
                        var searchText = $('#searchInput').val().toLowerCase();
                        var profitFilter = $('#profitFilter').val();
                        var minAccuracy = parseFloat($('#minAccuracy').val()) || 0;

                        $('#strategyTable tbody tr').each(function() {
                            var row = $(this);
                            var text = row.text().toLowerCase();
                            var profit = parseFloat(row.find('td:eq(6)').data('value'));
                            var accuracy = parseFloat(row.find('td:eq(9)').data('value'));
                            
                            var showRow = text.includes(searchText) && 
                                        accuracy >= minAccuracy &&
                                        (profitFilter === 'all' ||
                                         (profitFilter === 'profit' && profit > 0) ||
                                         (profitFilter === 'loss' && profit <= 0));
                            
                            row.toggle(showRow);
                        });
                    }
                });
            </script>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"\nResults exported to: {filepath}")
        print(f"Perfect profit (theoretical maximum): ${perfect_profit:.2f} with {perfect_trades} trades")
        print(f"Buy and Hold profit: ${buy_hold_profit:.2f}")

    def level1_raw(self, predictions: List[float]) -> List[float]:
        """Return first 3 raw predictions"""
        return predictions[:3]

    def level1_simple_average(self, predictions: List[float]) -> List[float]:
        """Calculate simple moving averages"""
        result = []
        for i in range(3):
            result.append(np.mean(predictions[i:i+5]))
        return result

    def level1_weighted_average(self, predictions: List[float], alpha: float = 0.5) -> List[float]:
        """Calculate weighted moving averages"""
        result = []
        for i in range(3):
            subset = predictions[i:i+5]
            n = len(subset)
            weights = [alpha * (1 - alpha) ** (n - 1 - j) for j in range(n)]
            total_weight = sum(weights)
            result.append(sum(p * w for p, w in zip(subset, weights)) / total_weight)
        return result

    def level1_exponential_average(self, predictions: List[float], alpha: float = 0.3) -> List[float]:
        """Calculate exponential moving average with more weight on recent predictions"""
        result = []
        for i in range(3):
            subset = predictions[i:i+5]
            weights = [alpha * (1-alpha)**(len(subset)-1-j) for j in range(len(subset))]
            weights = [w/sum(weights) for w in weights]  # Normalize weights
            result.append(sum(p * w for p, w in zip(subset, weights)))
        return result

    def level1_median_filter(self, predictions: List[float]) -> List[float]:
        """Use median instead of mean to reduce impact of outliers"""
        result = []
        for i in range(3):
            result.append(np.median(predictions[i:i+5]))
        return result

    def level2_simple_comparison(self, adjusted_values: List[float], close: float) -> int:
        """Simple comparison with close price"""
        return 1 if adjusted_values[0] > close else -1

    def level2_trend_analysis(self, adjusted_values: List[float], close: float) -> int:
        """Check for continuous increase/decrease"""
        if all(adjusted_values[i] < adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return 1
        if all(adjusted_values[i] > adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return -1
        return 0

    def level2_threshold(self, adjusted_values: List[float], close: float) -> int:
        """Check for 2% threshold"""
        percent_change = (adjusted_values[0] - close) / close * 100
        if percent_change >= 2:
            return 1
        if percent_change <= -2:
            return -1
        return 0

    def level2_combined(self, adjusted_values: List[float], close: float) -> int:
        """Combined trend and threshold analysis"""
        percent_change = (adjusted_values[0] - close) / close * 100
        if percent_change >= 2 and all(adjusted_values[i] < adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return 1
        if percent_change <= -2 and all(adjusted_values[i] > adjusted_values[i+1] for i in range(len(adjusted_values)-1)):
            return -1
        return 0

    def level2_momentum(self, adjusted_values: List[float], close: float) -> int:
        """Check price momentum using rate of change"""
        momentum = (adjusted_values[0] - adjusted_values[-1]) / adjusted_values[-1] * 100
        if momentum > 1.5:  # Strong upward momentum
            return 1
        if momentum < -1.5:  # Strong downward momentum
            return -1
        return 0

    def level2_volatility_based(self, adjusted_values: List[float], close: float) -> int:
        """Make decisions based on price volatility"""
        volatility = np.std(adjusted_values) / np.mean(adjusted_values) * 100
        price_change = (adjusted_values[0] - close) / close * 100
        
        if volatility < 1.0:  # Low volatility environment
            if price_change > 1.0:
                return 1
            if price_change < -1.0:
                return -1
        else:  # High volatility environment
            if price_change > 2.0:  # Require stronger signals in volatile markets
                return 1
            if price_change < -2.0:
                return -1
        return 0

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
        """Dynamically adjust weights based on ADX strength"""
        adx_strength = day.adx / 100  # Normalize ADX
        
        if adx_strength > 0.25:  # Strong trend
            # Give more weight to trend indicators
            rsi_weight = 0.4
            williams_weight = 0.4
            adx_weight = 0.2
        else:  # Weak trend
            # Give more weight to oscillators
            rsi_weight = 0.3
            williams_weight = 0.3
            adx_weight = 0.4
        
        rsi_factor = max(-0.5, min(0.5, (30 - day.rsi) / 100))
        williams_factor = max(-0.5, min(0.5, (-80 - day.williams) / 100))
        adx_factor = max(-0.5, min(0.5, (day.adx - 25) / 100))
        
        return (rsi_factor * rsi_weight + 
                williams_factor * williams_weight + 
                adx_factor * adx_weight)

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
        """Adaptive decision making based on signal strength"""
        if abs(score) < 0.2:  # Weak signal
            return 0
        elif abs(score) < 0.4:  # Moderate signal
            return 1 if score > 0 else -1
        else:  # Strong signal
            return 2 if score > 0 else -2  # Stronger position size

    def level4_trend_following(self, score: float) -> int:
        """More conservative in counter-trend moves"""
        if score > 0.3:  # Strong uptrend
            return 1
        elif score < -0.4:  # Require stronger signal for shorts
            return -1
        return 0

    def evaluate_prediction_method(self, method_name: str, predictions: List[float], day: TradingDay,
                                 level1_method: Callable, level2_method: Callable, 
                                 level3_method: Callable, level4_method: Callable) -> Tuple[int, float]:
        """Evaluate a single prediction using the specified methods"""
        adjusted_values = level1_method(predictions)
        signal = level2_method(adjusted_values, day.close)
        tech_adjustment = level3_method(day)
        combined_score = signal + tech_adjustment
        final_decision = level4_method(combined_score)
        return final_decision, combined_score

    def calculate_prediction_error(self, predictions: List[float], actual_prices: List[float]) -> float:
        """Calculate mean absolute error between predictions and actual prices"""
        if len(predictions) > len(actual_prices):
            predictions = predictions[:len(actual_prices)]
        errors = [abs(pred - actual) for pred, actual in zip(predictions, actual_prices)]
        return sum(errors) / len(errors)

    def calculate_optimal_weight(self, day1_pred: List[float], day2_pred: List[float], actual_prices: List[float]) -> WeightedPrediction:
        """Calculate optimal weight combination of two predictions to minimize error"""
        best_weight = 0.0
        min_error = float('inf')
        
        # Try different weight combinations from 0 to 1 in steps of 0.1
        for w in range(0, 11):
            weight = w / 10
            weighted_pred = [weight * p1 + (1 - weight) * p2 
                            for p1, p2 in zip(day1_pred, day2_pred)]
            
            # Calculate error for this weight combination
            errors = [abs(pred - actual) 
                     for pred, actual in zip(weighted_pred, actual_prices)]
            mean_error = sum(errors) / len(errors)
            
            if mean_error < min_error:
                min_error = mean_error
                best_weight = weight
        
        return WeightedPrediction(weight=best_weight, error=min_error)

    def evaluate_all_combinations(self):
        """Test all possible combinations of methods"""
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

        level1_methods = [
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

        level4_methods = [
            self.level4_conservative,
            self.level4_aggressive,
            self.level4_adaptive,
            self.level4_trend_following
        ]

        for method_name, get_predictions in prediction_methods:
            for l1 in level1_methods:
                for l2 in level2_methods:
                    for l3 in level3_methods:
                        for l4 in level4_methods:
                            result = self.backtest_strategy(
                                method_name, get_predictions, l1, l2, l3, l4
                            )
                            self.results.append(result)

        self.results.sort(key=lambda x: x.profit, reverse=True)

    def backtest_strategy(self, method_name: str, get_predictions: Callable,
                         level1_method: Callable, level2_method: Callable,
                         level3_method: Callable, level4_method: Callable) -> PredictionResult:
        """Backtest a strategy combination"""
        symbol_data = next(iter(self.trading_data.values()))
        position = 0
        entry_price = 0.0
        profit = 0.0
        total_trades = 0
        correct_trades = 0
        prediction_errors = []

        for i, day in enumerate(symbol_data[:-1]):
            predictions = get_predictions(day)
            decision, _ = self.evaluate_prediction_method(
                method_name, predictions, day,
                level1_method, level2_method,
                level3_method, level4_method
            )
            
            next_day = symbol_data[i + 1]
            
            # Calculate prediction error
            if i + len(predictions) < len(symbol_data):
                future_prices = [d.close for d in symbol_data[i+1:i+1+len(predictions)]]
                error = self.calculate_prediction_error(predictions, future_prices)
                prediction_errors.append(error)
            
            if decision == 1 and position == 0:
                position = 1
                entry_price = next_day.open
                total_trades += 1
            elif decision == -1 and position == 1:
                position = 0
                exit_price = next_day.open
                trade_profit = exit_price - entry_price
                profit += trade_profit
                if trade_profit > 0:
                    correct_trades += 1

        mean_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0

        return PredictionResult(
            method=method_name,
            level1_method=level1_method.__name__,
            level2_method=level2_method.__name__,
            level3_method=level3_method.__name__,
            level4_method=level4_method.__name__,
            profit=profit,
            total_trades=total_trades,
            correct_trades=correct_trades,
            mean_prediction_error=mean_prediction_error
        ) 

    def calculate_profit_percentage(self, profit: float, entry_price: float) -> float:
        """Calculate profit as a percentage of entry price"""
        return (profit / entry_price * 100) if entry_price != 0 else 0

    def calculate_buy_and_hold_profit(self) -> Tuple[float, float]:
        """Calculate profit and profit percentage from buying at start and holding until end"""
        symbol_data = next(iter(self.trading_data.values()))
        if len(symbol_data) < 2:
            return 0.0, 0.0
        
        entry_price = symbol_data[0].open
        exit_price = symbol_data[-1].close
        profit = exit_price - entry_price
        profit_percentage = self.calculate_profit_percentage(profit, entry_price)
        return profit, profit_percentage

    def calculate_perfect_profit(self) -> Tuple[float, float, int, List[Tuple[str, str, float]]]:
        """Calculate maximum possible profit with perfect future knowledge"""
        symbol_data = next(iter(self.trading_data.values()))
        max_profit = 0.0
        initial_investment = symbol_data[0].open  # Use first day's open price as initial investment
        position = 0
        entry_price = 0.0
        trade_count = 0
        trades = []  # To store trade details
        
        # Look for local minima to buy and local maxima to sell
        for i in range(1, len(symbol_data)-1):
            prev_day = symbol_data[i-1]
            curr_day = symbol_data[i]
            next_day = symbol_data[i+1]
            
            # Local minimum (buy signal)
            if curr_day.close < prev_day.close and curr_day.close < next_day.close and position == 0:
                position = 1
                entry_price = next_day.open
                trade_count += 1
                trades.append((next_day.date, 'BUY', entry_price))
            
            # Local maximum (sell signal)
            elif curr_day.close > prev_day.close and curr_day.close > next_day.close and position == 1:
                position = 0
                exit_price = next_day.open
                profit = exit_price - entry_price
                max_profit += profit
                trades.append((next_day.date, 'SELL', exit_price))
        
        profit_percentage = self.calculate_profit_percentage(max_profit, initial_investment)
        return max_profit, profit_percentage, trade_count, trades

    def get_best_strategy(self) -> PredictionResult:
        """Return the best performing strategy"""
        if not self.results:
            raise ValueError("No strategies evaluated yet. Run evaluate_all_combinations() first.")
        return self.results[0]

    def print_top_strategies(self, n: int = 5):
        """Print the top N performing strategies"""
        perfect_profit, perfect_percentage, perfect_trades, perfect_trade_details = self.calculate_perfect_profit()
        buy_hold_profit, buy_hold_percentage = self.calculate_buy_and_hold_profit()
        
        print("\nTrading Scenarios Comparison:")
        print(f"Perfect Trading Profit: ${perfect_profit:.2f} ({perfect_percentage:.1f}%) with {perfect_trades} trades")
        print(f"Buy and Hold Profit: ${buy_hold_profit:.2f} ({buy_hold_percentage:.1f}%) with 1 trade")
        print("\nPerfect Trading Details:")
        for date, action, price in perfect_trade_details:
            print(f"{date}: {action} at ${price:.2f}")
        
        print(f"\nTop {n} Performing Strategies:")
        for i, result in enumerate(self.results[:n], 1):
            profit_percentage = (result.profit/perfect_profit*100) if perfect_profit > 0 else 0
            initial_investment = next(iter(self.trading_data.values()))[0].open
            strategy_percentage = self.calculate_profit_percentage(result.profit, initial_investment)
            print(f"\n{i}. {result}")
            print(f"   Return: {strategy_percentage:.1f}%")
            print(f"   Percentage of Perfect Profit: {profit_percentage:.1f}%") 