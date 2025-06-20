import pandas as pd
from dask.distributed import Client, LocalCluster
import time
import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.backtester import Backtester, BacktestResult
from core.optimizer import SignalBasedOptimizer
from core.sma_strategy import SMAStrategy, SMAParams

def run_basic_backtest(start_line: int = None):
    """Runs a basic backtest for the SMA Crossover strategy with default parameters."""
    print("\n=== Running Basic SMA Crossover Backtest ===")
    
    backtester = Backtester()
    
    
    # Create strategy with default or specified parameters
    basic_params = SMAParams(short_window=10, long_window=150)
    sma_strategy = SMAStrategy(basic_params)
    
    print(f"Strategy: {sma_strategy.get_name()}")
    print(f"Description: {sma_strategy.get_description()}")

    fixed_backtest_params = {
        'initial_capital': 100.0, # 100 USD
        'fee_pct': 0.025,
        'position_sizing_pct': 0.25, # 25% of capital per trade (must be decimal, not percentage)
        'max_trades_per_coin': 1, # Keeps it simple, but could be changed to a higher number
        'execution_delay_seconds': 3, # Simple slippage simulation
        'flat_fee_usd': 0.0,  # Match optimizer: explicit flat fee
        'stop_loss_pct': 20.0, # Example stop loss
        'scramble_order': False,  # Match optimizer: explicit scramble setting
        'start_line': start_line
    }

    result = backtester.backtest_strategy(
        strategy=sma_strategy,
        **fixed_backtest_params
    )
    
    result.print_results()
    result.plot_results()
    return result

def run_parameter_optimization():
    """Optimizes the SMA Crossover strategy parameters using a local Dask cluster."""
    print("\n=== Running SMA Crossover Parameter Optimization (using Dask) ===")
    
    backtester = Backtester()
    
    # Strategy template for the optimizer to use
    sma_strategy_template = SMAStrategy(SMAParams())

    # Define the parameter ranges for the optimizer to test
    param_ranges = {
        'short_window': (10, 51, 10),     # Test short windows from 10 to 50, step 10
        'long_window': (60, 151, 30),     # Test long windows from 60 to 150, step 30
        'stop_loss_pct': (5.0, 20.1, 5.0), # Optimize the backtester's stop loss (stop loss is processed separately from the strategy)
    }
    
    # Fixed parameters for all backtest runs during optimization
    fixed_backtest_params = {
        'initial_capital': 100.0,
        'fee_pct': 0.025,
        'flat_fee_usd': 0.0,
        'position_sizing_pct': 0.25,  # Match single backtest: 25% of capital per trade
        'max_trades_per_coin': 1,
        'execution_delay_seconds': 3,
        'use_walk_forward': False,
        'train_size': 1.0,
        'stop_loss_pct': None # Default if 'stop_loss_pct' is NOT in param_ranges
    }

    optimizer = SignalBasedOptimizer(
        parameter_ranges=param_ranges,
        use_walk_forward=fixed_backtest_params['use_walk_forward'],
        train_size=fixed_backtest_params['train_size']
    )
    
    client = None
    cluster = None
    try:
        print("Creating Local Dask Cluster...")
        cluster = LocalCluster()
        client = Client(cluster)
        print(f"Dask dashboard link: {client.dashboard_link}")

        print("Preloading data and scattering to workers...")
        backtester.preload_price_data()
        backtester_future = client.scatter(backtester, broadcast=True)
        strategy_future = client.scatter(sma_strategy_template, broadcast=True)

        print("Uploading required Python modules to workers...")
        # Create a package structure on workers to ensure imports work
        client.run(lambda: os.makedirs('core', exist_ok=True))
        client.run(lambda: open("core/__init__.py", "w").close())
        # Upload all necessary files
        client.upload_file("core/backtester.py")
        client.upload_file("core/optimizer.py")
        client.upload_file("core/result.py")
        client.upload_file("core/strategies.py")
        client.upload_file("core/sma_strategy.py") # The new strategy

        print("Starting optimization. This may take some time...")
        best_summary_dict = optimizer.optimize(
            client=client, # Pass the active client
            backtester_future=backtester_future, # Pass the future
            strategy_future=strategy_future, # Pass the future
            strategy_template=sma_strategy_template, # Pass the template for local use
            initial_capital=fixed_backtest_params['initial_capital'],
            fee_pct=fixed_backtest_params['fee_pct'],
            position_sizing_pct=fixed_backtest_params['position_sizing_pct'],
            max_trades_per_coin=fixed_backtest_params['max_trades_per_coin'],
            flat_fee_usd=fixed_backtest_params['flat_fee_usd'],
            stop_loss_pct=fixed_backtest_params['stop_loss_pct'], # Default/fallback
            execution_delay_seconds=fixed_backtest_params['execution_delay_seconds'],
            scramble_order=False, # Explicitly disable scrambling for deterministic results
            random_sample_size=None
        )

    except Exception as e:
        print(f"An error occurred during Dask optimization: {e}")
        best_summary_dict = None
    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()
        print("Dask resources closed.")

    print("\n=== Optimization Complete ===")
    if best_summary_dict:
        print("Best Parameters & Corresponding Results:")
        for key, value in best_summary_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        return best_summary_dict
    else:
        print("\nOptimization did not yield a best result or may have failed.")
        return None

if __name__ == "__main__":
    # Uncomment the function you want to run
    
    # Option 1: Run a single backtest with specific parameters
    #run_basic_backtest(start_line=None)

    # Option 2: Run the optimizer to find the best parameters
    best_params = run_parameter_optimization() 