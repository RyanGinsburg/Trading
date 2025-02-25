import json
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import numpy as np  # type: ignore
import os
from datetime import datetime

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
        accuracy = (self.correct_trades / self.total_trades * 100) if self.total_trades > 0 else 0
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
        # Ensure optimal_weights is defined before loading data
        self.optimal_weights = {}  # symbol -> list of optimal weights dict per group
        self.trading_data = self._load_data(json_file)
        # We'll store per-stock and global buy-and-hold metrics here.
        self.buy_hold_by_stock = {}  # symbol -> (total_profit, percent_profit)
        self.global_buy_hold = None

    def _load_data(self, json_file: str) -> Dict[str, List[List[TradingDay]]]:
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
                group_data = group['data']
                trading_days = []
                group_optimal_weights = {}
                # First pass: calculate optimal weights per prediction pair for this group
                for i, entry in enumerate(group_data):
                    predictions = {}
                    for key, value in entry.items():
                        if key.startswith('pred_'):
                            float_list = [float(x) for x in value.split(',')]
                            if float_list:
                                float_list = float_list[1:]  # Remove the first number
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
                            float_list = [float(x) for x in value.split(',')]
                            if float_list:
                                float_list = float_list[1:]
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

    def calculate_optimal_weight(self, day1_pred: List[float], day2_pred: List[float], actual_prices: List[float]) -> WeightedPrediction:
        best_weight = 0.0
        min_error = float('inf')
        for w in range(0, 11):
            weight = w / 10
            weighted_pred = [weight * p1 + (1 - weight) * p2
                             for p1, p2 in zip(day1_pred, day2_pred)]
            errors = [abs(pred - actual)
                      for pred, actual in zip(weighted_pred, actual_prices)]
            mean_error = sum(errors) / len(errors)
            if mean_error < min_error:
                min_error = mean_error
                best_weight = weight
        return WeightedPrediction(weight=best_weight, error=min_error)

    def backtest_strategy_group(self, group: List[TradingDay], method_name: str, get_predictions: Callable,
                                  level1_method: Callable, level2_method: Callable,
                                  level3_method: Callable, level4_method: Callable) -> PredictionResult:
        position = 0
        entry_price = 0.0
        profit = 0.0
        total_trades = 0
        correct_trades = 0
        prediction_errors = []
        n = len(group)
        for i, day in enumerate(group[:-1]):
            predictions = get_predictions(day)
            decision, _ = self.evaluate_prediction_method(
                method_name, predictions, day,
                level1_method, level2_method,
                level3_method, level4_method
            )
            next_day = group[i + 1]
            if i + len(predictions) < n:
                future_prices = [d.close for d in group[i+1:i+1+len(predictions)]]
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

    def evaluate_all_combinations_group(self, group: List[TradingDay]) -> List[PredictionResult]:
        results = []
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
        count = 0
        total_combinations = len(prediction_methods) * len(level1_methods) * len(level2_methods) * len(level3_methods) * len(level4_methods)
        for method_name, get_predictions in prediction_methods:
            for l1 in level1_methods:
                for l2 in level2_methods:
                    for l3 in level3_methods:
                        for l4 in level4_methods:
                            count += 1
                            if count % 1000 == 0:
                                print(f"      Processed {count} / {total_combinations} strategy combinations for this group...")
                            result = self.backtest_strategy_group(group, method_name, get_predictions, l1, l2, l3, l4)
                            results.append(result)
        return results

    def _calculate_buy_and_hold_profit(self, group: List[TradingDay]) -> Tuple[float, float]:
        if len(group) < 2:
            return 0.0, 0.0
        entry_price = group[0].open
        exit_price = group[-1].close
        profit = exit_price - entry_price
        profit_percentage = (profit / entry_price * 100) if entry_price != 0 else 0
        return profit, profit_percentage

    def _calculate_perfect_profit(self, group: List[TradingDay]) -> Tuple[float, float, int, List[Tuple[str, str, float]]]:
        """
        Computes the maximum possible profit using next-day open prices.
        It sums every positive difference between consecutive open prices.
        """
        max_profit = 0.0
        trades = []
        n = len(group)
        for i in range(n - 1):
            diff = group[i + 1].open - group[i].open
            if diff > 0:
                max_profit += diff
                trades.append((group[i + 1].date, 'TRADE', diff))
        initial_investment = group[0].open
        profit_pct = (max_profit / initial_investment * 100) if initial_investment != 0 else 0
        return max_profit, profit_pct, len(trades), trades


    def run_all_backtests(self):
        """
        For each stock and each group within that stock, evaluate all strategy combinations.
        For each group, also compute the buy-and-hold profit.
        Then, accumulate per-stock cumulative strategy stats and compute overall best strategy.
        Additionally, compute cumulative buy-and-hold performance per stock and globally.
        """
        self.results_by_stock = {}  # symbol -> list of tuples: (group_index, best_strategy, perfect_result, all_results, group_initial, bh_profit, bh_pct)
        self.cumulative_results_by_stock = {}  # symbol -> { strategy_key: cumulative stats }
        global_initial = 0.0
        global_profit = 0.0
        for symbol, groups in self.trading_data.items():
            print(f"Processing stock: {symbol} with {len(groups)} groups...")
            self.results_by_stock[symbol] = []
            cumulative = {}
            total_initial_buy_hold = 0.0
            total_profit_buy_hold = 0.0
            for group_index, group in enumerate(groups):
                if not group:
                    print(f"  Warning: Group {group_index+1} for stock {symbol} is empty. Skipping.")
                    continue
                print(f"  Processing group {group_index+1}/{len(groups)} for stock {symbol}...")
                results = self.evaluate_all_combinations_group(group)
                print(f"    Completed evaluation for group {group_index+1} ({len(results)} strategies tested)")
                best_strategy = max(results, key=lambda r: (r.profit / group[0].open * 100) if group[0].open != 0 else -float('inf'))
                perfect_result = self._calculate_perfect_profit(group)
                bh_profit, bh_pct = self._calculate_buy_and_hold_profit(group)
                self.results_by_stock[symbol].append((group_index, best_strategy, perfect_result, results, group[0].open, bh_profit, bh_pct))
                for res in results:
                    key = (res.method, res.level1_method, res.level2_method, res.level3_method, res.level4_method)
                    if key not in cumulative:
                        cumulative[key] = {
                            'total_profit': 0.0,
                            'total_initial': 0.0,
                            'total_trades': 0,
                            'total_correct': 0
                        }
                    cumulative[key]['total_profit'] += res.profit
                    cumulative[key]['total_initial'] += group[0].open
                    cumulative[key]['total_trades'] += res.total_trades
                    cumulative[key]['total_correct'] += res.correct_trades
                total_initial_buy_hold += group[0].open
                profit_bh = group[-1].close - group[0].open
                total_profit_buy_hold += profit_bh
            self.cumulative_results_by_stock[symbol] = cumulative
            self.buy_hold_by_stock[symbol] = (total_profit_buy_hold, (total_profit_buy_hold / total_initial_buy_hold * 100) if total_initial_buy_hold != 0 else 0)
            global_initial += total_initial_buy_hold
            global_profit += total_profit_buy_hold
            print(f"Completed processing stock: {symbol}")
        # Compute global buy-and-hold performance
        self.global_buy_hold = (global_profit, (global_profit / global_initial * 100) if global_initial != 0 else 0)
        # Determine best cumulative strategy per stock
        self.best_strategy_by_stock = {}
        for symbol, cum in self.cumulative_results_by_stock.items():
            best_key = None
            best_pct = -float('inf')
            for key, stats in cum.items():
                initial = stats['total_initial']
                profit_pct = (stats['total_profit'] / initial * 100) if initial != 0 else 0
                if profit_pct > best_pct:
                    best_pct = profit_pct
                    best_key = key
            if best_key:
                self.best_strategy_by_stock[symbol] = (best_key, best_pct, cum[best_key])
        # Determine global best strategy across all stocks
        global_cumulative = {}
        for symbol, cum in self.cumulative_results_by_stock.items():
            for key, stats in cum.items():
                if key not in global_cumulative:
                    global_cumulative[key] = {
                        'total_profit': 0.0,
                        'total_initial': 0.0,
                        'total_trades': 0,
                        'total_correct': 0
                    }
                global_cumulative[key]['total_profit'] += stats['total_profit']
                global_cumulative[key]['total_initial'] += stats['total_initial']
                global_cumulative[key]['total_trades'] += stats['total_trades']
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
            self.global_best = (best_global_key, best_global_pct, global_cumulative[best_global_key])
        else:
            self.global_best = None

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

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .stock-section {{ margin-bottom: 40px; padding: 10px; border: 1px solid #ddd; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
                th {{ background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading Strategy Results</h1>
        """
        for symbol, group_results in self.results_by_stock.items():
            bh_profit_stock, bh_pct_stock = self.buy_hold_by_stock.get(symbol, (0,0))
            html_content += f"<div class='stock-section'><h2>Stock: {symbol}</h2>"
            for (group_index, best_strategy, perfect_result, _, group_initial, group_bh_profit, group_bh_pct) in group_results:
                best_pct = (best_strategy.profit / group_initial * 100) if group_initial != 0 else 0
                perfect_profit, perfect_pct, perfect_trades, _ = perfect_result
                # Build a table for each group with best strategy, its % profit, buy-and-hold % and perfect trading %.
                html_content += f"""
                <h3>Group {group_index+1}</h3>
                <table>
                    <tr>
                        <th>Best Strategy</th>
                        <th>Profit ($)</th>
                        <th>% Profit</th>
                        <th>Total Trades</th>
                    </tr>
                    <tr>
                        <td>
                            Prediction: {best_strategy.method}<br>
                            L1: {best_strategy.level1_method}<br>
                            L2: {best_strategy.level2_method}<br>
                            L3: {best_strategy.level3_method}<br>
                            L4: {best_strategy.level4_method}
                        </td>
                        <td>${best_strategy.profit:.2f}</td>
                        <td>{best_pct:.2f}%</td>
                        <td>{best_strategy.total_trades}</td>
                    </tr>
                </table>
                <p><strong>Buy and Hold:</strong> Profit: ${group_bh_profit:.2f}, % Profit: {group_bh_pct:.2f}%</p>
                <p><strong>Perfect Trading:</strong> Profit: ${perfect_profit:.2f}, % Profit: {perfect_pct:.2f}%, Trades: {perfect_trades}</p>
                <hr>
                """
            html_content += f"""
                <h3>Cumulative for {symbol}</h3>
                <p><strong>Best Cumulative Strategy:</strong> Prediction {self.best_strategy_by_stock[symbol][0][0]}, L1: {self.best_strategy_by_stock[symbol][0][1]}, L2: {self.best_strategy_by_stock[symbol][0][2]}, L3: {self.best_strategy_by_stock[symbol][0][3]}, L4: {self.best_strategy_by_stock[symbol][0][4]}</p>
                <p><strong>Cumulative Profit Percentage:</strong> {self.best_strategy_by_stock[symbol][1]:.2f}%</p>
                <p><strong>Cumulative Buy and Hold:</strong> Profit: ${bh_profit_stock:.2f}, % Profit: {bh_pct_stock:.2f}%</p>
            </div>
            """
        if self.global_best:
            best_key, best_global_pct, _ = self.global_best
            global_bh_profit, global_bh_pct = self.global_buy_hold
            html_content += f"""
            <div class='stock-section'>
                <h2>Global Best Strategy</h2>
                <p>Strategy: Prediction {best_key[0]}, L1: {best_key[1]}, L2: {best_key[2]}, L3: {best_key[3]}, L4: {best_key[4]}</p>
                <p>Global Cumulative Profit Percentage: {best_global_pct:.2f}%</p>
                <p><strong>Global Buy and Hold:</strong> Profit: ${global_bh_profit:.2f}, % Profit: {global_bh_pct:.2f}%</p>
            </div>
            """
        html_content += """
            </div>
        </body>
        </html>
        """
        with open(filepath, 'w') as f:
            f.write(html_content)
        print(f"Results exported to: {filepath}")
