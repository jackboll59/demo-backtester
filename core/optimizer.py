import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dask.distributed import Client, Future, progress as dask_progress
from core.strategies import Strategy
from itertools import product
from core.result import PlottingMixin
from scipy import stats
from tqdm import tqdm

if TYPE_CHECKING:
    from core.backtester import Backtester

class SignalBasedOptimizer:
    """Class to optimize strategy parameters based on buy/sell signals using Dask"""
    def __init__(self, 
                 parameter_ranges: Dict[str, Tuple[float, float, float]] = None,
                 use_walk_forward: bool = False,
                 train_size: float = 0.7):
        """
        Initialize the optimizer with parameter ranges to test.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to (start, end, step) tuples
            use_walk_forward: Whether to use walk-forward optimization (train/test split)
            train_size: Proportion of data to use for training (default 0.7 or 70%)
        """
        self.parameter_ranges = parameter_ranges or {}
        self.use_walk_forward = use_walk_forward
        self.train_size = train_size
        self.tqdm = tqdm
            
    def _split_watch_data(self, watch_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split watch data into training and testing sets chronologically."""
        # Sort by watch start time to ensure chronological order
        sorted_data = watch_data.sort_values('watch_start_time')
        
        # Calculate split index
        split_idx = int(len(sorted_data) * self.train_size)
        
        # Split the data
        train_data = sorted_data.iloc[:split_idx].copy()
        test_data = sorted_data.iloc[split_idx:].copy()
        
        return train_data, test_data
    
    def _run_backtest(self, params_tuple: Tuple, # The varying parameters for this run
                      # --- Fixed arguments start here ---
                      backtester: 'Backtester',  # Use forward reference
                      strategy_template: Strategy,
                      initial_capital: float,
                      fee_pct: float,
                      position_sizing_pct: float,
                      max_trades_per_coin: Optional[int],
                      max_trade_duration_minutes: Optional[float],
                      max_watch_for_entry_minutes: Optional[float],
                      optimized_param_names: List[str],
                      flat_fee_usd: float,
                      default_stop_loss_pct: Optional[float] = None,
                      execution_delay_seconds: int = 3,
                      scramble_order: bool = False,
                      ) -> Optional[Dict]:
        """
        Run a single backtest with the given parameters. Designed to be called by Dask.
        Returns ONLY a dictionary containing raw metrics and parameters used.
        """
        try:
            # Create mapping from param name to value for this run
            current_run_params = dict(zip(optimized_param_names, params_tuple))

            # Determine the stop loss for this specific run
            current_stop_loss = current_run_params.get('stop_loss_pct', default_stop_loss_pct)

            # Start with default params from the template strategy's parameter dataclass
            strategy_params_dict = strategy_template.params.__dict__.copy()

            # Override defaults with parameters being optimized in this run
            strategy_override_params = {k: v for k, v in current_run_params.items() if k != 'stop_loss_pct'}
            strategy_params_dict.update(strategy_override_params)

            # Create a copy of the strategy with modified parameters
            strategy_class = strategy_template.__class__
            params_class = strategy_template.params.__class__
            current_params_obj = params_class(**strategy_params_dict)
            current_strategy = strategy_class(current_params_obj)

            # Run backtest with current parameters
            result = backtester.backtest_strategy(
                strategy=current_strategy,
                flat_fee_usd=flat_fee_usd,
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                position_sizing_pct=position_sizing_pct,
                max_trades_per_coin=max_trades_per_coin,
                max_trade_duration_minutes=max_trade_duration_minutes,
                max_watch_for_entry_minutes=max_watch_for_entry_minutes,
                stop_loss_pct=current_stop_loss,
                execution_delay_seconds=execution_delay_seconds,
                scramble_order=scramble_order
            )

            # Return raw metrics without scoring
            return_dict = {
                'total_return_pct': result.total_portfolio_return_pct,
                'max_drawdown_pct': result.max_drawdown,
                'win_rate_pct': result.win_rate * 100,
                'winning_trades': result.winning_trades,
                'executed_trades': result.total_executed_trades,
                'trades': result.total_executed_trades,
                'run_error': False,
            }

            # Add strategy parameters
            for param_name, param_value in current_params_obj.__dict__.items():
                if param_name == 'stop_loss_pct' and 'stop_loss_pct' in current_run_params:
                    continue
                return_dict[param_name] = param_value

            # Add backtester stop loss if it was optimized
            if 'stop_loss_pct' in current_run_params:
                return_dict['stop_loss_pct'] = current_stop_loss

            return return_dict

        except Exception as e:
            print(f"[Worker Error] Exception during backtest for params {params_tuple}: {e}")
            error_dict = {'run_error': True, 'error_message': str(e)}
            try:
                error_dict.update(dict(zip(optimized_param_names, params_tuple)))
            except:
                error_dict['params_tuple'] = params_tuple
            return error_dict

    def _calculate_scores(self, results: List[Dict], backtester: 'Backtester') -> List[Dict]:
        """
        Calculate scores for all results, balancing total return against max drawdown.
        Outliers are not penalized, and the focus is on the return/drawdown ratio.
        """
        if not results:
            return []

        # Use a small minimum for drawdown to handle cases of zero or near-zero drawdown gracefully.
        # This prevents division-by-zero errors and disproportionately high scores for tiny drawdowns.
        MINIMUM_DRAWDOWN = 0.1 # Represents 0.1%

        for result in results:
            # If the run failed, assign a very low score.
            if result.get('run_error', False):
                result['score'] = float('-inf')
                continue

            return_pct = result.get('total_return_pct', 0.0)
            drawdown_pct = abs(result.get('max_drawdown_pct', 0.0))

            # Heavily penalize any strategy that loses money.
            if return_pct <= 0:
                # The score will be a large negative number, ensuring it's ranked last.
                result['score'] = return_pct - 1000
            else:
                # Calculate the effective drawdown.
                effective_drawdown = max(drawdown_pct, MINIMUM_DRAWDOWN)
                
                # The score is the ratio of return to effective drawdown.
                result['score'] = return_pct / effective_drawdown

        return results

    def optimize(self,
                 client: Client,
                 backtester_future: Future,
                 strategy_future: Future,
                 strategy_template: Strategy,
                 initial_capital: float = 100.0,
                 fee_pct: float = 0.025,
                 position_sizing_pct: float = 0.25,
                 max_trades_per_coin: int = None,
                 max_trade_duration_minutes: float = None,
                 max_watch_for_entry_minutes: float = None,
                 flat_fee_usd: float = 0.0,
                 stop_loss_pct: float = None,
                 execution_delay_seconds: int = 3,
                 scramble_order: bool = False,
                 random_sample_size: Optional[int] = None
                 ) -> Optional[Dict]:
        """
        Find optimal parameter combination for the strategy using a provided Dask client and scattered data.
        If walk-forward optimization is enabled, will split data into training/testing sets.
        
        Args:
            client: Dask client to use for distributed computation
            backtester_future: Future object containing the backtester
            strategy_future: Future object containing the strategy
            strategy_template: Strategy template to use for backtesting
            initial_capital: Initial capital for backtesting
            fee_pct: Fee percentage for backtesting
            position_sizing_pct: Position sizing percentage for backtesting
            max_trades_per_coin: Maximum trades per coin for backtesting
            max_trade_duration_minutes: Maximum trade duration in minutes for backtesting
            max_watch_for_entry_minutes: Maximum watch duration in minutes for entry detection
            flat_fee_usd: Flat fee in USD for backtesting
            stop_loss_pct: Stop loss percentage for backtesting
            execution_delay_seconds: Execution delay in seconds for backtesting
            scramble_order: If True, randomly shuffle the order of watch sessions processed. Defaults to False.
            random_sample_size: If set, runs optimization on a random sample of N combinations. Defaults to None (run all).
        """
        # Generate parameter combinations
        param_iterables = []
        optimized_param_names = []

        for param_name, param_range in self.parameter_ranges.items():
            start, end, step = param_range
            if step == 0:
                if start != end:
                     print(f"Warning: Step is 0 for parameter '{param_name}'. Using only start value {start}.")
                param_iterables.append([start])
            else:
                epsilon = step / 1000.0 
                param_iterables.append(np.arange(start, end + epsilon, step))
                
            optimized_param_names.append(param_name)
        
        all_param_combinations = list(product(*param_iterables))
        
        # this is custom code for dip_bounce strat that filters out nonsensical combinations
        if 'dip_threshold' in optimized_param_names and 'max_dip_pct' in optimized_param_names:
            dip_threshold_idx = optimized_param_names.index('dip_threshold')
            max_dip_pct_idx = optimized_param_names.index('max_dip_pct')
            
            original_count = len(all_param_combinations)
            all_param_combinations = [
                p for p in all_param_combinations if p[dip_threshold_idx] < p[max_dip_pct_idx]
            ]
            filtered_count = original_count - len(all_param_combinations)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} combinations where dip_threshold >= max_dip_pct.")

        total_combinations = len(all_param_combinations)
        
        # --- Add random sampling logic ---
        if random_sample_size is not None and random_sample_size < total_combinations:
            print(f"\n--- Running Random Sample Optimization ---")
            print(f"Selecting a random sample of {random_sample_size} combinations from a total of {total_combinations}.")
            all_param_combinations = random.sample(all_param_combinations, random_sample_size)
            total_combinations = len(all_param_combinations) # Update the count
        
        if total_combinations == 0:
            print("No parameter combinations generated. Check parameter ranges.")
            return None
        
        print(f"\nOptimizing over {total_combinations} combinations for parameters: {optimized_param_names}...")
        
        if total_combinations > 1:
             proceed = input(f"About to run {total_combinations} backtest combinations. Proceed? (y/n): ").strip().lower()
             if proceed != 'y':
                 print("Optimization aborted by user.")
                 return None
             print("Proceeding with optimization...")

        try:
            # Get the backtester object to calculate target trades
            backtester = client.gather(backtester_future)
            num_coins = len(backtester.watch_data['watch_id'].unique())
            target_trades = num_coins * 0.5  # Target is 50% of available coins
            trade_std = num_coins * 0.25     # Allow wider variance for larger datasets
            print(f"\nTrade Count Target: {target_trades:.0f} trades (±{trade_std:.0f})")
            print(f"Based on {num_coins} unique coins in dataset")

            if self.use_walk_forward:
                print(f"\nUsing Walk-Forward Optimization")
                print(f"Training Set Size: {self.train_size*100:.1f}%")
                print(f"Test Set Size: {(1-self.train_size)*100:.1f}%")
                
                train_data, test_data = self._split_watch_data(backtester.watch_data)
                
                # Create new backtester instances for train and test
                train_backtester = backtester.__class__()
                test_backtester = backtester.__class__()
                
                # Set the watch_data for each
                train_backtester.watch_data = train_data
                test_backtester.watch_data = test_data
                
                # Scatter the new backtesters
                train_backtester_future = client.scatter(train_backtester, broadcast=True)
                test_backtester_future = client.scatter(test_backtester, broadcast=True)
            else:
                train_backtester_future = backtester_future
                test_backtester_future = None

            # Run all backtests at once
            fixed_args_for_map = dict(
                backtester=train_backtester_future,
                strategy_template=strategy_future,
                initial_capital=initial_capital,
                fee_pct=fee_pct,
                position_sizing_pct=position_sizing_pct,
                max_trades_per_coin=max_trades_per_coin,
                max_trade_duration_minutes=max_trade_duration_minutes,
                max_watch_for_entry_minutes=max_watch_for_entry_minutes,
                optimized_param_names=optimized_param_names,
                flat_fee_usd=flat_fee_usd,
                default_stop_loss_pct=stop_loss_pct,
                execution_delay_seconds=execution_delay_seconds,
                scramble_order=scramble_order,
            )
            
            print("\nRunning backtests...")
            futures: List[Future] = client.map(self._run_backtest, all_param_combinations, **fixed_args_for_map)
            
            print("Waiting for results...")
            dask_progress(futures)

            # Gather all results
            results = []
            try:
                batch_results = client.gather(futures, errors='raise')
                results.extend([r for r in batch_results if not r.get('run_error', False)])
            except Exception as e:
                print(f"Error gathering results: {e}")

            if not results:
                print("No valid results generated during optimization.")
                return None

            # Calculate scores for all results
            scored_results = self._calculate_scores(results, backtester)
            
            # Find best result
            best_result = max(scored_results, key=lambda x: x.get('score', float('-inf')))

            # If using walk-forward, test the best parameters
            if self.use_walk_forward and best_result:
                print("\nTesting best parameters on test set...")
                test_args = fixed_args_for_map.copy()
                test_args.update({
                    'backtester': test_backtester_future,
                })
                
                # Run single backtest with best params on test set
                test_future = client.submit(self._run_backtest, 
                                         tuple(best_result[p] for p in optimized_param_names),
                                         **test_args)
                test_result = client.gather(test_future)
                
                if not test_result.get('run_error', False):
                    print("\nTest Set Performance:")
                    print(f"Total Return: {test_result['total_return_pct']:+.2f}%")
                    print(f"Max Drawdown: {test_result['max_drawdown_pct']:.2f}%")
                    print(f"Win Rate: {test_result['win_rate_pct']:.2f}%")
                    print(f"Total Trades: {test_result['trades']}")
                    
                    # Add test results to best_result
                    best_result.update({
                        'test_return_pct': test_result['total_return_pct'],
                        'test_drawdown_pct': test_result['max_drawdown_pct'],
                        'test_win_rate_pct': test_result['win_rate_pct'],
                        'test_trades': test_result['trades']
                    })

            # Create results DataFrame for plotting
            results_df = pd.DataFrame(scored_results)
            
            # Format results for plotting
            formatters = {
                'total_return_pct': '{:+.2f}%'.format,
                'max_drawdown_pct': '{:.2f}%'.format,
                'win_rate_pct': '{:.2f}%'.format,
                'score': '{:.2f}'.format
            }
            
            for col, fmt in formatters.items():
                if col in results_df.columns:
                    results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                    results_df[col] = results_df[col].apply(lambda x: fmt(x) if pd.notna(x) else 'NaN')

            results_df['score_numeric'] = pd.to_numeric(results_df['score'], errors='coerce').fillna(float('-inf'))
            
            # Plot results
            PlottingMixin.plot_optimization_results(results_df, best_result, self.parameter_ranges)
            
            return best_result

        except Exception as e:
            print(f"An critical error occurred during optimization: {e}")
            return None
