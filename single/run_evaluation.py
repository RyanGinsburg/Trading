import os
import webbrowser
from datetime import datetime
from eval.trading_strategy import StrategyTester

def print_top_strategies(tester, n):
    """
    For each stock, print the top n groups’ best strategies.
    Each entry in tester.results_by_stock[symbol] is a tuple:
      (group_index, best_strategy, perfect_result, all_results, group_initial, bh_profit, bh_pct)
    """
    for symbol, results in tester.results_by_stock.items():
        print(f"\nStock: {symbol}")
        for tup in results[:n]:
            # Unpack all 7 elements:
            group_index, res, perfect, _, group_initial, group_bh_profit, group_bh_pct = tup
            accuracy = (res.correct_trades / res.total_trades * 100) if res.total_trades > 0 else 0
            best_pct = (res.profit / group_initial * 100) if group_initial != 0 else 0
            perfect_profit, perfect_pct, perfect_trades, _ = perfect
            print(f"Group {group_index+1}:")
            print(f"  Prediction Source: {res.method}")
            print(f"  Level1: {res.level1_method}")
            print(f"  Level2: {res.level2_method}")
            print(f"  Level3: {res.level3_method}")
            print(f"  Level4: {res.level4_method}")
            print(f"  Strategy Profit: ${res.profit:.2f} ({best_pct:.2f}%) with {res.total_trades} trades, Accuracy: {accuracy:.2f}%")
            print(f"  Buy and Hold Profit: ${group_bh_profit:.2f} ({group_bh_pct:.2f}%)")
            print(f"  Perfect Trading Profit: ${perfect_profit:.2f} ({perfect_pct:.2f}%) with {perfect_trades} trades")
            print("-" * 40)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Updated JSON file name to 'trading_database_export.json'
    json_path = os.path.join(current_dir, 'json', 'trading_database_export.json')
    print("Initializing StrategyTester...")
    tester = StrategyTester(json_path)
    print("Evaluating all strategy combinations for all stocks and groups...")
    tester.run_all_backtests()
    print_top_strategies(tester, 10)
    if tester.global_best:
        best_key, best_pct, stats = tester.global_best
        global_bh_profit, global_bh_pct = tester.global_buy_hold
        print("\nGlobal Best Strategy Details:")
        print(f"Strategy: Prediction {best_key[0]}, L1: {best_key[1]}, L2: {best_key[2]}, L3: {best_key[3]}, L4: {best_key[4]}")
        print(f"Global Cumulative Profit Percentage: {best_pct:.2f}%")
        print(f"Global Buy and Hold Profit: ${global_bh_profit:.2f} ({global_bh_pct:.2f}%)")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strategy_results_{timestamp}.html"
    tester.export_results_to_html(filename)
    results_path = os.path.join(current_dir, "..", "eval", "results", filename)
    webbrowser.open("file://" + os.path.realpath(results_path))

if __name__ == "__main__":
    main()
