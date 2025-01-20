import os
import webbrowser
from datetime import datetime
from eval.trading_strategy import StrategyTester

def main():
    # Get the directory where run_evaluation.py is (single/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to single directory
    json_path = os.path.join(current_dir, 'json', 'trading_database_export.json')
    
    # Create strategy tester
    tester = StrategyTester(json_path)
    
    # Run evaluation
    print("Evaluating all strategy combinations...")
    tester.evaluate_all_combinations()
    
    # Print results
    tester.print_top_strategies(10)
    
    best_strategy = tester.get_best_strategy()
    print("\nBest Strategy Details:")
    print(best_strategy)
    
    # Export and automatically open results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'strategy_results_{timestamp}.html'
    tester.export_results_to_html(filename)
    
    # Open the HTML file in the default browser
    results_path = os.path.join(current_dir, 'eval', 'results', filename)
    webbrowser.open('file://' + os.path.realpath(results_path))

if __name__ == "__main__":
    main()