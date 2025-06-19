import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from itertools import product
from core.strategies import Strategy, StrategyParams
from collections import defaultdict
import random # Import random for shuffling
import time # Import time for estimation
import os # Import os for cpu count
# --- Dask Imports ---
from dask.distributed import Client, LocalCluster, Future, progress as dask_progress # Added Future and progress
# --- End Dask Imports ---
from core.result import PlottingMixin
import numba

# --- Numba Optimized Helper Functions ---
@numba.jit(nopython=True)
def _process_signals_and_stops_nb(
    price_timestamps_ns: np.ndarray,
    prices: np.ndarray,
    signal_timestamps_ns: np.ndarray,
    signal_types: np.ndarray,  # 1 for buy, -1 for sell
    signal_prices: np.ndarray,
    execution_delay_ns: int,
    stop_loss_pct: float,
    max_trade_duration_ns: int,
    max_watch_for_entry_ns: int,
    watch_start_time_ns: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized function to process signals and stop losses.
    
    Returns:
        - trade_entry_times_ns: Entry timestamps in nanoseconds
        - trade_entry_prices: Entry prices
        - trade_exit_times_ns: Exit timestamps in nanoseconds  
        - trade_exit_prices: Exit prices
        - trade_triggers: Trigger types (0=strategy_signal, 1=stop_loss, 2=max_duration, 3=session_end)
        - trade_signal_times_ns: Original signal timestamps
        - trade_signal_prices: Original signal prices
        - final_position_entry_time_ns: If position still open, entry time (-1 if not)
        - final_position_entry_price: If position still open, entry price (0.0 if not)
    """
    n_prices = len(prices)
    n_signals = len(signal_timestamps_ns)
    
    # Pre-allocate trade arrays (max possible trades is n_signals // 2)
    max_trades = max(n_signals, 1)
    trade_entry_times_ns = np.empty(max_trades, dtype=np.int64)
    trade_entry_prices = np.empty(max_trades, dtype=np.float64)
    trade_exit_times_ns = np.empty(max_trades, dtype=np.int64)
    trade_exit_prices = np.empty(max_trades, dtype=np.float64)
    trade_triggers = np.empty(max_trades, dtype=np.int32)
    trade_signal_times_ns = np.empty(max_trades, dtype=np.int64)
    trade_signal_prices = np.empty(max_trades, dtype=np.float64)
    
    trade_count = 0
    current_signal_idx = 0
    
    # Position tracking
    in_position = False
    position_entry_time_ns = 0
    position_entry_price = 0.0
    position_signal_time_ns = 0
    position_signal_price = 0.0
    
    # Helper function to find price at timestamp
    def find_price_at_time(target_time_ns):
        # Binary search for the closest timestamp >= target_time_ns
        left, right = 0, n_prices - 1
        while left <= right:
            mid = (left + right) // 2
            if price_timestamps_ns[mid] < target_time_ns:
                left = mid + 1
            else:
                right = mid - 1
        
        if left >= n_prices:
            return n_prices - 1  # Return last available index
        return left
    
    # Process each price point
    for i in range(n_prices):
        current_time_ns = price_timestamps_ns[i]
        current_price = prices[i]
        
        # --- Check for Stop Loss ---
        if in_position and stop_loss_pct != 0.0:
            stop_price_level = position_entry_price * (1.0 - stop_loss_pct / 100.0)
            
            if current_price <= stop_price_level:
                # Stop loss triggered
                execution_time_ns = current_time_ns + execution_delay_ns
                execution_idx = find_price_at_time(execution_time_ns)
                
                exit_time_ns = price_timestamps_ns[execution_idx]
                exit_price = prices[execution_idx]
                trigger = 1  # stop_loss
                
                # Check max duration
                if max_trade_duration_ns > 0:
                    max_duration_time_ns = position_entry_time_ns + max_trade_duration_ns
                    if exit_time_ns > max_duration_time_ns:
                        duration_exit_idx = find_price_at_time(max_duration_time_ns)
                        exit_time_ns = price_timestamps_ns[duration_exit_idx]
                        exit_price = prices[duration_exit_idx]
                        trigger = 2  # max_duration
                
                # Record trade
                trade_entry_times_ns[trade_count] = position_entry_time_ns
                trade_entry_prices[trade_count] = position_entry_price
                trade_exit_times_ns[trade_count] = exit_time_ns
                trade_exit_prices[trade_count] = exit_price
                trade_triggers[trade_count] = trigger
                trade_signal_times_ns[trade_count] = position_signal_time_ns
                trade_signal_prices[trade_count] = position_signal_price
                trade_count += 1
                
                in_position = False
                continue
        
        # --- Process Strategy Signals ---
        while (current_signal_idx < n_signals and 
               signal_timestamps_ns[current_signal_idx] == current_time_ns):
            
            signal_type = signal_types[current_signal_idx]
            signal_price = signal_prices[current_signal_idx]
            signal_time_ns = signal_timestamps_ns[current_signal_idx]
            current_signal_idx += 1
            
            # Apply execution delay
            execution_time_ns = signal_time_ns + execution_delay_ns
            execution_idx = find_price_at_time(execution_time_ns)
            
            if execution_idx >= n_prices:
                continue  # Skip if delay pushes past available data
                
            actual_execution_time_ns = price_timestamps_ns[execution_idx]
            actual_execution_price = prices[execution_idx]
            
            # Handle buy signal
            if signal_type == 1 and not in_position:  # buy
                # Check max watch time
                if max_watch_for_entry_ns > 0:
                    time_since_watch_start_ns = signal_time_ns - watch_start_time_ns
                    if time_since_watch_start_ns > max_watch_for_entry_ns:
                        continue
                
                in_position = True
                position_entry_time_ns = actual_execution_time_ns
                position_entry_price = actual_execution_price
                position_signal_time_ns = signal_time_ns
                position_signal_price = signal_price
                
            # Handle sell signal
            elif signal_type == -1 and in_position:  # sell
                exit_time_ns = actual_execution_time_ns
                exit_price = actual_execution_price
                trigger = 0  # strategy_signal
                
                # Check max duration
                if max_trade_duration_ns > 0:
                    max_duration_time_ns = position_entry_time_ns + max_trade_duration_ns
                    if exit_time_ns > max_duration_time_ns:
                        duration_exit_idx = find_price_at_time(max_duration_time_ns)
                        exit_time_ns = price_timestamps_ns[duration_exit_idx]
                        exit_price = prices[duration_exit_idx]
                        trigger = 2  # max_duration
                
                # Record trade
                trade_entry_times_ns[trade_count] = position_entry_time_ns
                trade_entry_prices[trade_count] = position_entry_price
                trade_exit_times_ns[trade_count] = exit_time_ns
                trade_exit_prices[trade_count] = exit_price
                trade_triggers[trade_count] = trigger
                trade_signal_times_ns[trade_count] = position_signal_time_ns
                trade_signal_prices[trade_count] = position_signal_price
                trade_count += 1
                
                in_position = False
                break  # Break signal processing for this timestamp
    
    # Return final position state
    final_position_entry_time_ns = position_entry_time_ns if in_position else -1
    final_position_entry_price = position_entry_price if in_position else 0.0
    
    return (
        trade_entry_times_ns[:trade_count],
        trade_entry_prices[:trade_count],
        trade_exit_times_ns[:trade_count],
        trade_exit_prices[:trade_count],
        trade_triggers[:trade_count],
        trade_signal_times_ns[:trade_count],
        trade_signal_prices[:trade_count],
        final_position_entry_time_ns,
        final_position_entry_price
    )

class BacktestResult(PlottingMixin):
    """Class to hold and analyze backtest results"""
    def __init__(self, trades: List[Dict], watch_data: pd.DataFrame, initial_capital: float = 100.0,
                 fee_pct: float = 0.025, flat_fee_usd: float = 0.0, 
                 position_sizing_pct: float = 0.25):
        self.trades = trades
        self.watch_data = watch_data
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct # Fee per side (entry/exit)
        self.flat_fee_usd = flat_fee_usd # Flat fee per side (entry/exit)
        self.position_sizing_pct = position_sizing_pct # Percentage of current equity per trade
        
        # Initialize fee accumulators
        self.total_pct_fees_usd = 0.0
        self.total_flat_fees_usd = 0.0
        
        # Calculate metrics
        self.total_trades = len(trades) # Total signals generated
        
        # Early return if no trades
        if self.total_trades == 0:
            self._initialize_empty_results()
            return
            
        # Calculate returns and wallet value over time dynamically
        self.wallet_values = [initial_capital]
        current_value = initial_capital
        
        # Track actual returns considering position size and fees
        actual_returns_pct_vs_initial = [] 
        executed_trade_indices = [] # Keep track of which trades actually executed
        
        # Prepare metadata analysis containers
        self.trade_durations = []  # Time between entry and exit
        self.entry_delays = []     # Time between watch start and entry
        self.trade_ages = []       # Age of coin at trade time
        self.trade_volumes = []    # Volume at trade time
        self.trade_mcaps = []      # Market cap at trade time
        self.trade_liquidities = [] # Liquidity at trade time
        self.trade_metadata = []    # Store complete metadata for each trade
        
        for i, trade in enumerate(trades):
            # Get metadata for this trade
            watch_id = trade['watch_id']
            watch_info = watch_data[watch_data['watch_id'] == watch_id].iloc[0]
            
            # Calculate timing metrics
            watch_start = pd.to_datetime(watch_info['watch_start_time'])
            watch_end = pd.to_datetime(watch_info['watch_end_time'])
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            trade_duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
            entry_delay = (entry_time - watch_start).total_seconds() / 60   # in minutes
            
            # Store metadata
            trade_metadata = {
                'duration': trade_duration,
                'entry_delay': entry_delay,
                'age': watch_info['age'],
                'volume': watch_info['vol'],
                'mcap': watch_info['mcap'],
                'liquidity': watch_info['liq'],
                'watch_duration': (watch_end - watch_start).total_seconds() / 60
            }
            self.trade_metadata.append(trade_metadata)
            
            # Calculate position size based on current equity
            current_position_size = current_value * self.position_sizing_pct
            if current_position_size <= 0: # Cannot trade if equity is zero or negative
                 continue # Skip this trade signal

            executed_trade_indices.append(i) # Mark trade as executed
            
            # Store metadata for executed trades
            self.trade_durations.append(trade_duration)
            self.entry_delays.append(entry_delay)
            self.trade_ages.append(watch_info['age'])
            self.trade_volumes.append(watch_info['vol'])
            self.trade_mcaps.append(watch_info['mcap'])
            self.trade_liquidities.append(watch_info['liq'])

            # Calculate profit/loss in USD before fees
            # Ensure entry price is positive for calculation
            if trade['entry_price'] <= 0: continue
            profit_pct = trade['profit_pct']
            profit_usd_gross = current_position_size * (profit_pct / 100)
            
            # Calculate fees
            entry_value = current_position_size
            # Exit value must be non-negative for fee calculation
            exit_value = max(0, current_position_size + profit_usd_gross)
            entry_fee = entry_value * (self.fee_pct / 100)
            exit_fee = exit_value * (self.fee_pct / 100)
            pct_fee_component = entry_fee + exit_fee
            flat_fee_component = self.flat_fee_usd * 2  # Both entry and exit
            # Combine percentage and flat fees
            total_fees = pct_fee_component + flat_fee_component
            
            # Accumulate fee components
            self.total_pct_fees_usd += pct_fee_component
            self.total_flat_fees_usd += flat_fee_component
            
            # Calculate net profit/loss in USD
            profit_usd_net = profit_usd_gross - total_fees
            
            # Update current wallet value
            current_value += profit_usd_net
            # Ensure wallet value doesn't go below zero (due to fees/losses)
            current_value = max(0, current_value) 
            self.wallet_values.append(current_value)

            # Store net return percentage relative to initial capital for consistency with other metrics
            actual_return_pct = (profit_usd_net / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            actual_returns_pct_vs_initial.append(actual_return_pct)
            
            # Add net profit and metadata to trade dict
            trade['net_profit_usd'] = profit_usd_net
            trade['total_fees_usd'] = total_fees
            trade['position_size_usd'] = current_position_size
            trade.update(trade_metadata)  # Add metadata to trade dict

        # Create list of trades that were actually executed
        self.executed_trades = [self.trades[i] for i in executed_trade_indices]
        self.total_executed_trades = len(self.executed_trades)

        # Update win rate based on executed trades and net profit > 0
        self.winning_trades = len([t for t in self.executed_trades if t.get('net_profit_usd', 0) > 0])
        self.win_rate = self.winning_trades / self.total_executed_trades if self.total_executed_trades > 0 else 0

        self.final_wallet_value = self.wallet_values[-1] if len(self.wallet_values) > 1 else initial_capital
        # Total return calculation remains based on final vs initial capital
        self.total_portfolio_return_pct = ((self.final_wallet_value - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        # Calculate average *trade* return (net of fees) relative to initial capital 
        self.avg_return_pct = np.mean(actual_returns_pct_vs_initial) if actual_returns_pct_vs_initial else 0
        
        # Calculate Max Drawdown correctly using portfolio value history
        wallet_values_arr = np.array(self.wallet_values)
        if len(wallet_values_arr) > 1:
            # Calculate running peak (maximum value seen so far)
            peaks = np.maximum.accumulate(wallet_values_arr)
            # Prevent division by zero if peak is 0 
            valid_peaks = np.where(peaks <= 0, 1e-10, peaks)  # Use small positive value instead of 0
            # Calculate drawdowns as percentage
            drawdowns = (wallet_values_arr - peaks) / valid_peaks
            # Get the maximum drawdown (a negative percentage)
            self.max_drawdown = float(np.min(drawdowns) * 100) if len(drawdowns) > 0 else 0
        else:
            self.max_drawdown = 0.0

        # Calculate Sharpe-like ratio using the actual net returns relative to initial capital
        returns_array = np.array(actual_returns_pct_vs_initial)
        std_dev = np.std(returns_array)
        self.risk_adjusted_return = np.mean(returns_array) / std_dev if len(returns_array) > 0 and std_dev != 0 else 0
            
        # Calculate per-coin metrics based on executed trades and gross profit %
        self.coin_metrics = self._calculate_coin_metrics(self.executed_trades)
        
    def _initialize_empty_results(self):
        """Initialize empty or default values when there are no trades"""
        self.wallet_values = [self.initial_capital]
        self.final_wallet_value = self.initial_capital
        self.total_portfolio_return_pct = 0.0
        self.avg_return_pct = 0.0
        self.max_drawdown = 0.0
        self.risk_adjusted_return = 0.0
        self.winning_trades = 0
        self.total_executed_trades = 0
        self.win_rate = 0.0
        self.executed_trades = []
        self.trade_durations = []
        self.entry_delays = []
        self.trade_ages = []
        self.trade_volumes = []
        self.trade_mcaps = []
        self.trade_liquidities = []
        self.trade_metadata = []
        self.coin_metrics = {}
        print("\nWarning: No trades were executed in the selected range.")
        
    def _calculate_coin_metrics(self, trades_to_analyze: List[Dict]) -> Dict:
        """Calculate performance metrics per coin based on provided trades"""
        coin_data = defaultdict(list)
        for trade in trades_to_analyze:
            watch_id = trade['watch_id']
            coin_data[watch_id].append(trade)
            
        metrics = {}
        for watch_id, trades in coin_data.items():
            num_trades = len(trades)
            if num_trades == 0: continue 
            
            # Calculate metrics based on gross profit_pct for consistency with previous intent
            win_rate_gross = len([t for t in trades if t['profit_pct'] > 0]) / num_trades 
            avg_return_gross = np.mean([t['profit_pct'] for t in trades])
            total_return_gross = sum(t['profit_pct'] for t in trades)
            max_drawdown_gross = min([t['profit_pct'] for t in trades if 'profit_pct' in t], default=0)
            # Calculate total net profit per coin
            total_net_profit_usd = sum(t.get('net_profit_usd', 0) for t in trades)

            metrics[watch_id] = {
                'total_trades': num_trades,
                'win_rate_gross': win_rate_gross, 
                'avg_return_gross': avg_return_gross, 
                'total_return_gross': total_return_gross,
                'max_drawdown_gross': max_drawdown_gross,
                'total_net_profit_usd': total_net_profit_usd
            }
        return metrics

    def print_results(self):
        """Print backtest results with proper handling of edge cases"""
        print("\n" + "="*40)
        print("      BACKTEST RESULTS SUMMARY")
        print("="*40)
        print(f"Execution Parameters: Fee={self.fee_pct:.3f}%/side ({self.fee_pct*2:.3f}% R/T), Pos. Size={self.position_sizing_pct*100:.1f}% of Equity")
        if self.flat_fee_usd > 0:
            print(f"                      Flat Fee=${self.flat_fee_usd:.2f}/side (${self.flat_fee_usd*2:.2f} R/T)")
        print("-"*40)
        
        if self.total_trades == 0:
            print("No trades were executed in the selected range.")
            print("="*40 + "\n")
            return
            
        print("Portfolio Performance:")
        print(f"  Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"  Final Portfolio Value: ${self.final_wallet_value:,.2f}")
        print(f"  Total Net Return:      {self.total_portfolio_return_pct:+.2f}%")
        print(f"  Max Drawdown:          {self.max_drawdown:.2f}%")
        
        # Only print risk-adjusted return if we have enough trades
        if len(self.executed_trades) > 1:
            print(f"  Risk-Adjusted Return:  {self.risk_adjusted_return:.2f} (Avg Net Trade Ret / StDev Net Trade Ret)")
        else:
            print("  Risk-Adjusted Return:  N/A (Insufficient trades)")
            
        print("-"*40)
        print("Trade Stats:")
        print(f"  Total Signals Generated: {self.total_trades}")
        print(f"  Total Trades Executed: {self.total_executed_trades} (Skipped {self.total_trades - self.total_executed_trades})")
        print(f"  Win Rate (Net Profit>0):{self.win_rate*100:.2f}% ({self.winning_trades} Wins / {self.total_executed_trades - self.winning_trades} Losses)")
        
        # Add Trade Duration Analysis
        if self.trade_durations:
            print("-"*40)
            print("Trade Duration Analysis:")
            durations = np.array(self.trade_durations)
            print(f"  Average Duration:     {np.mean(durations):.2f} minutes")
            print(f"  Median Duration:      {np.median(durations):.2f} minutes")
            print(f"  Min/Max Duration:     {np.min(durations):.2f} / {np.max(durations):.2f} minutes")
            
            # Group trades by duration buckets
            duration_buckets = [
                (0, 1, "<1 min"),
                (1, 2, "1-2 min"),
                (2, 5, "2-5 min"),
                (5, 8, "5-8 min"),
                (8, 10, "8-10 min")
            ]
            
            print("  Duration Distribution:")
            for low, high, label in duration_buckets:
                count = sum(1 for d in durations if low <= d < high)
                pct = (count / len(durations)) * 100 if len(durations) > 0 else 0
                print(f"    {label}: {count} trades ({pct:.1f}%)")
                
            # Win rate by duration
            if self.executed_trades:
                print("  Win Rate by Duration:")
                for low, high, label in duration_buckets:
                    bucket_trades = [t for t in self.executed_trades if low <= t.get('duration', 0) < high]
                    if bucket_trades:
                        winners = sum(1 for t in bucket_trades if t.get('net_profit_usd', 0) > 0)
                        win_rate = (winners / len(bucket_trades)) * 100
                        print(f"    {label}: {win_rate:.1f}% ({winners}/{len(bucket_trades)})")
        
        # Session End Analysis
        if self.executed_trades:
            print("-"*40)
            print("Session End Analysis:")
            session_end_trades = [t for t in self.executed_trades if t.get('trigger_type') == 'session_end']
            signal_trades = [t for t in self.executed_trades if t.get('trigger_type') == 'strategy_signal']
            
            session_end_count = len(session_end_trades)
            session_end_pct = (session_end_count / self.total_executed_trades) * 100 if self.total_executed_trades > 0 else 0
            print(f"  Trades closed by signals: {len(signal_trades)} ({100-session_end_pct:.1f}%)")
            print(f"  Trades cut off by session end: {session_end_count} ({session_end_pct:.1f}%)")
            
            # Win rate comparison
            if session_end_trades:
                session_end_winners = sum(1 for t in session_end_trades if t.get('net_profit_usd', 0) > 0)
                session_end_win_rate = (session_end_winners / session_end_count) * 100
                print(f"  Session end trade win rate: {session_end_win_rate:.1f}% ({session_end_winners}/{session_end_count})")
            
            if signal_trades:
                signal_winners = sum(1 for t in signal_trades if t.get('net_profit_usd', 0) > 0)
                signal_win_rate = (signal_winners / len(signal_trades)) * 100
                print(f"  Signal-based trade win rate: {signal_win_rate:.1f}% ({signal_winners}/{len(signal_trades)})")
        
        # Entry Timing Analysis
        if self.entry_delays:
            print("-"*40)
            print("Entry Timing Analysis:")
            delays = np.array(self.entry_delays)
            print(f"  Average entry delay: {np.mean(delays):.2f} minutes")
            print(f"  Median entry delay:  {np.median(delays):.2f} minutes")
            
            # Group entries by timing buckets
            entry_buckets = [
                (0, 1, "<1 min"),
                (1, 2, "1-2 min"),
                (2, 4, "2-4 min"),
                (4, 7, "4-7 min"),
                (7, 10, ">7 min")
            ]
            
            print("  Entry Delay Distribution:")
            for low, high, label in entry_buckets:
                count = sum(1 for d in delays if low <= d < high)
                pct = (count / len(delays)) * 100 if len(delays) > 0 else 0
                print(f"    {label}: {count} entries ({pct:.1f}%)")
                
            # Win rate by entry timing
            if self.executed_trades:
                print("  Win Rate by Entry Timing:")
                for low, high, label in entry_buckets:
                    bucket_trades = [t for t in self.executed_trades if low <= t.get('entry_delay', 0) < high]
                    if bucket_trades:
                        winners = sum(1 for t in bucket_trades if t.get('net_profit_usd', 0) > 0)
                        win_rate = (winners / len(bucket_trades)) * 100
                        print(f"    {label}: {win_rate:.1f}% ({winners}/{len(bucket_trades)})")
        
        print("-"*40)
        print("Top 5 Coins by Total Net Profit (USD):") 
        # Sort by total_net_profit_usd
        sorted_coins = sorted(self.coin_metrics.items(), 
                            key=lambda x: x[1]['total_net_profit_usd'], 
                            reverse=True)[:5]
        if not sorted_coins:
             print("  No coin data available (no executed trades).")
        for coin_id, metrics in sorted_coins:
            print(f"  Coin {coin_id[:15]}...")
            print(f"    Net Profit: ${metrics['total_net_profit_usd']:,.2f} ({metrics['total_trades']} trades)")
            # Handle potential NaN in gross metrics
            win_rate_str = f"{metrics['win_rate_gross']*100:.1f}%" if pd.notna(metrics.get('win_rate_gross')) else "N/A"
            avg_ret_str = f"{metrics['avg_return_gross']:.2f}%" if pd.notna(metrics.get('avg_return_gross')) else "N/A"
            print(f"    Gross Win Rate: {win_rate_str}, Avg Gross Ret: {avg_ret_str}")
        print("-"*40)
        print("Fee Summary (Executed Trades):")
        print(f"  Total Percentage Fees: ${self.total_pct_fees_usd:,.2f}")
        if self.flat_fee_usd > 0:
            print(f"  Total Flat Fees:       ${self.total_flat_fees_usd:,.2f}")
            print(f"  Grand Total Fees:    ${self.total_pct_fees_usd + self.total_flat_fees_usd:,.2f}")
        print("="*40 + "\n")

class Backtester:
    def __init__(self):
        # Load and validate watch data
        try:
            self.watch_data = pd.read_csv('data/watch_tracking.csv')
            if 'watch_id' not in self.watch_data.columns:
                raise ValueError("'watch_id' column missing in watch_tracking.csv")
            self.watch_data['watch_start_time'] = pd.to_datetime(self.watch_data['watch_start_time'])
            self.watch_data['watch_end_time'] = pd.to_datetime(self.watch_data['watch_end_time'])
        except FileNotFoundError:
            print("Error: watch_tracking.csv not found in data directory")
            raise
        except Exception as e:
            print(f"Error loading/processing watch data: {e}")
            raise
            
        # We'll load price data on demand rather than all at once
        self.price_data_cache = {}  # Cache to store loaded price data by watch_id
        
        # Preload unique watch_ids for faster lookup
        self.unique_watch_ids = set(self.watch_data['watch_id'].unique())
        
        # Verify price data file exists and has correct columns
        try:
            price_data_sample = pd.read_csv('data/price_history.csv', nrows=1)
            required_cols = ['watch_id', 'price', 'timestamp']
            if not all(col in price_data_sample.columns for col in required_cols):
                raise ValueError("Required columns missing in price_history.csv")
        except FileNotFoundError:
            print("Error: price_history.csv not found")
            raise
            
    def preload_price_data(self):
        """Preload all price data in a single pass - use for faster optimization runs"""
        print("Preloading price data for all watch sessions...")
        chunk_size = 50000
        
        # Try to use parquet first
        parquet_loaded = False
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
            
            try:
                dataset = ds.dataset('price_history.parquet')
                for watch_id in self.unique_watch_ids:
                    try:
                        # Filter by watch_id and select columns
                        table = dataset.scanner(
                            filter=ds.field('watch_id') == watch_id, 
                            columns=['timestamp', 'price']
                        ).to_table()
                        
                        # Convert to pandas
                        df = table.to_pandas()
                        if len(df) > 0:
                            # Convert timestamp and sort
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.sort_values('timestamp')
                            # Handle duplicate timestamps by keeping the first occurrence
                            df = df.drop_duplicates(subset=['timestamp'], keep='first')
                            self.price_data_cache[watch_id] = df
                    except Exception as watch_error:
                        print(f"Error loading watch_id {watch_id} from parquet: {watch_error}")
            except Exception as e:
                print(f"Error loading parquet dataset: {e}")
            
            parquet_loaded = True
            print(f"Preloaded price data from parquet for {len(self.price_data_cache)} watch sessions")
        
        except FileNotFoundError:
            print("price_history.parquet not found, falling back to CSV...")
            
        except ImportError:
            print("pyarrow not available, falling back to CSV...")
        except Exception as e:
            print(f"Unexpected error with parquet loading: {e}")
        
        # If parquet didn't work, try CSV
        if not parquet_loaded:
            try:
                # Process CSV file in chunks
                for chunk in pd.read_csv('price_history.csv', chunksize=chunk_size):
                    # Filter to only include watch_ids we care about
                    filtered_chunk = chunk[chunk['watch_id'].isin(self.unique_watch_ids)].copy()
                    
                    # Group by watch_id and add to cache
                    for watch_id, group in filtered_chunk.groupby('watch_id'):
                        if watch_id in self.price_data_cache:
                            self.price_data_cache[watch_id] = pd.concat([self.price_data_cache[watch_id], group])
                        else:
                            self.price_data_cache[watch_id] = group
                
                # Process all watch_ids in the cache
                for watch_id in list(self.price_data_cache.keys()):
                    try:
                        # Make sure we have a copy before modifying
                        df = self.price_data_cache[watch_id].copy()
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Drop duplicate timestamps, keeping the first occurrence
                        df = df.drop_duplicates(subset=['timestamp'], keep='first')
                        df = df.sort_values('timestamp')
                        self.price_data_cache[watch_id] = df
                    except Exception as e:
                        print(f"Error processing watch_id {watch_id}: {e}")
                        # Remove problematic watch_id from cache
                        self.price_data_cache.pop(watch_id, None)
                
                print(f"Preloaded price data from CSV for {len(self.price_data_cache)} watch sessions")
                
            except FileNotFoundError:
                print("Error: price_history.csv not found.")
            except Exception as csv_error:
                print(f"Error loading CSV data: {csv_error}")
            
        # If no data loaded, ensure cache is empty to avoid partial data issues
        if len(self.price_data_cache) == 0:
            print("No price data was loaded. Data will be loaded on demand.")
            self.price_data_cache = {}
    
    def get_price_history(self, watch_id: str) -> pd.DataFrame:
        """Get price history for a specific watch session, with caching"""
        # Return from cache if already loaded
        if watch_id in self.price_data_cache:
            return self.price_data_cache[watch_id]
        
        prices = None
        
        try:
            # First try parquet
            try:
                # Load only the necessary columns and filter by watch_id directly
                prices = pd.read_parquet(
                    'price_history.parquet',
                    columns=['timestamp', 'price'],
                    filters=[('watch_id', '==', watch_id)]
                )
            except FileNotFoundError:
                # Fall back to CSV if parquet not found
                try:
                    # Filter CSV by watch_id
                    prices = pd.read_csv(
                        'price_history.csv',
                        usecols=['watch_id', 'timestamp', 'price']
                    )
                    prices = prices[prices['watch_id'] == watch_id].copy()
                except FileNotFoundError:
                    print(f"Error: Neither price_history.parquet nor price_history.csv found for {watch_id}.")
                    return pd.DataFrame(columns=['timestamp', 'price'])
            
            if prices is None or len(prices) == 0:
                return pd.DataFrame(columns=['timestamp', 'price'])
            
            prices['timestamp'] = pd.to_datetime(prices['timestamp'])
            prices = prices.drop_duplicates(subset=['timestamp'], keep='first').sort_values('timestamp')
            self.price_data_cache[watch_id] = prices
            return prices
        except Exception as e:
            print(f"Error loading price data for watch_id {watch_id}: {e}")
            return pd.DataFrame(columns=['timestamp', 'price'])

    def backtest_strategy(self, strategy: Strategy, 
                         initial_capital: float = 100.0,
                         execution_delay_seconds: int = 3, 
                         max_trades_per_coin: int = None,
                         fee_pct: float = 0.025, 
                         position_sizing_pct: float = 0.25,
                         max_trade_duration_minutes: float = None,
                         max_watch_for_entry_minutes: float = None,
                         flat_fee_usd: float = 0.0,
                         stop_loss_pct: float = None,
                         scramble_order: bool = False,
                         start_line: int = None) -> BacktestResult:
        """
        Backtest a strategy across all watch sessions with fees and dynamic sizing.
        
        Args:
            strategy: Strategy instance to test
            initial_capital: Starting capital for the backtest
            execution_delay_seconds: Simulated delay for trade execution
            max_trades_per_coin: Maximum number of trades allowed per coin (None for unlimited)
            fee_pct: Trading fee percentage per side (entry/exit)
            position_sizing_pct: Percentage of current equity to use per trade
            max_trade_duration_minutes: Maximum duration for a trade in minutes (None for no limit)
            max_watch_for_entry_minutes: Maximum time to wait for an entry signal after watch start (None for no limit)
            flat_fee_usd: Flat fee in USD applied per trade side (entry/exit), defaults to 0.0
            stop_loss_pct: Optional stop-loss percentage based on actual entry price. 
                           Overrides any internal strategy stop-loss. Defaults to None (disabled).
            scramble_order: If True, randomly shuffle the order of watch sessions processed. Defaults to False.
            start_line: Line number in watch_tracking.csv to start from (1-based indexing). Defaults to None (start from beginning).
        Returns:
            BacktestResult object containing detailed results.
        """
        trades = [] # List to store details of each trade
        trades_per_coin = defaultdict(int)  # Track number of trades generated per coin
        
        # --- Determine iteration order and starting point --- 
        watch_data_indices = self.watch_data.index.tolist()
        
        if scramble_order:
            print("--- Scramble Mode: Watch session order randomized ---")
            random.shuffle(watch_data_indices)
            
        # Apply start_line filter after potential scrambling
        if start_line is not None:
            start_idx = start_line - 1
            if start_idx < 0 or start_idx >= len(watch_data_indices):
                raise ValueError(f"start_line {start_line} out of range (1-{len(watch_data_indices)})")
            watch_data_indices = watch_data_indices[start_idx:]
            print(f"Starting from line {start_line} (index {start_idx})")

        # Process watch sessions
        for idx in watch_data_indices:
            watch = self.watch_data.loc[idx]
            
            watch_id = watch['watch_id']
            
            # Skip if we've generated max trades for this coin
            if max_trades_per_coin and trades_per_coin[watch_id] >= max_trades_per_coin:
                continue
                
            # Get price data for this watch session
            prices = self.get_price_history(watch_id)
            if len(prices) < 2:
                continue
                
            # Get signals from strategy
            try:
                signals = strategy.generate_signals(prices)
                if not signals:
                    continue
            except Exception as e:
                print(f"Strategy {strategy.__class__.__name__} failed for {watch_id}: {e}")
                continue
                
            # Sort signals chronologically
            signals = sorted(signals, key=lambda s: pd.to_datetime(s[2]))
            
            # Prepare data for numba function
            price_timestamps_ns = prices['timestamp'].astype('datetime64[ns]').astype(np.int64).values
            price_values = prices['price'].values
            
            # Convert signals to numpy arrays
            if signals:
                signal_timestamps_ns = np.array([pd.to_datetime(s[2]).value for s in signals], dtype=np.int64)
                signal_types = np.array([1 if s[0] == 'buy' else -1 for s in signals], dtype=np.int32)
                signal_prices_arr = np.array([s[1] for s in signals], dtype=np.float64)
                
                # Sort signals by timestamp
                sort_idx = np.argsort(signal_timestamps_ns)
                signal_timestamps_ns = signal_timestamps_ns[sort_idx]
                signal_types = signal_types[sort_idx]
                signal_prices_arr = signal_prices_arr[sort_idx]
            else:
                signal_timestamps_ns = np.array([], dtype=np.int64)
                signal_types = np.array([], dtype=np.int32)
                signal_prices_arr = np.array([], dtype=np.float64)
            
            # Convert parameters for numba function
            execution_delay_ns = execution_delay_seconds * 1_000_000_000
            max_trade_duration_ns = int(max_trade_duration_minutes * 60 * 1_000_000_000) if max_trade_duration_minutes else 0
            max_watch_for_entry_ns = int(max_watch_for_entry_minutes * 60 * 1_000_000_000) if max_watch_for_entry_minutes else 0
            watch_start_time_ns = watch['watch_start_time'].value
            stop_loss_pct_val = stop_loss_pct or 0.0
            
            # Call numba function
            (trade_entry_times_ns, trade_entry_prices, trade_exit_times_ns, trade_exit_prices, 
             trade_triggers, trade_signal_times_ns, trade_signal_prices, 
             final_position_entry_time_ns, final_position_entry_price) = _process_signals_and_stops_nb(
                price_timestamps_ns, price_values, signal_timestamps_ns, signal_types, signal_prices_arr,
                execution_delay_ns, stop_loss_pct_val, max_trade_duration_ns, max_watch_for_entry_ns, watch_start_time_ns
            )
            
            # Convert results back to trade records
            trigger_names = ['strategy_signal', 'stop_loss', 'max_duration', 'session_end']
            
            for i in range(len(trade_entry_times_ns)):
                # Check max_trades_per_coin before adding trade
                if max_trades_per_coin and trades_per_coin[watch_id] >= max_trades_per_coin:
                    break
                    
                # Calculate profit
                entry_price = trade_entry_prices[i]
                exit_price = trade_exit_prices[i]
                profit_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                
                trade = {
                    'watch_id': watch_id,
                    'entry_price': entry_price,
                    'entry_time': pd.to_datetime(trade_entry_times_ns[i]),
                    'exit_price': exit_price,
                    'exit_time': pd.to_datetime(trade_exit_times_ns[i]),
                    'profit_pct': profit_pct,
                    'trigger_type': trigger_names[trade_triggers[i]],
                    'execution_delay': execution_delay_seconds,
                    'signal_price': trade_signal_prices[i],
                    'signal_time': pd.to_datetime(trade_signal_times_ns[i])
                }
                trades.append(trade)
                trades_per_coin[watch_id] += 1
            
            # Handle final open position
            if final_position_entry_time_ns != -1:
                open_position = {
                    'entry_price': final_position_entry_price,
                    'entry_time': pd.to_datetime(final_position_entry_time_ns),
                    'signal_price': final_position_entry_price,  # Approximation
                    'signal_time': pd.to_datetime(final_position_entry_time_ns)  # Approximation
                }
            else:
                open_position = None 
            
            # --- Session End Handling --- 
            # If position is still open after processing signals
            if open_position is not None:
                # Check max_trades_per_coin before adding session end trade
                if not max_trades_per_coin or trades_per_coin[watch_id] < max_trades_per_coin:
                    # Find the last price point
                    if not prices.empty:
                        last_timestamp = prices.iloc[-1]['timestamp']
                        last_price = prices.iloc[-1]['price']
                        exit_price = last_price
                        exit_time = last_timestamp
                        trigger = 'session_end'
                        
                        # Check if max trade duration forces an earlier exit
                        if max_trade_duration_minutes is not None:
                            max_duration_time = open_position['entry_time'] + pd.Timedelta(minutes=max_trade_duration_minutes)
                            
                            if exit_time > max_duration_time:
                                # Find price at max duration time
                                duration_mask = prices['timestamp'] <= max_duration_time
                                if duration_mask.any():
                                    duration_prices = prices[duration_mask]
                                    if not duration_prices.empty:
                                        exit_time = duration_prices.iloc[-1]['timestamp']
                                        exit_price = duration_prices.iloc[-1]['price']
                                        trigger = 'max_duration'

                        # Calculate profit percentage
                        if open_position['entry_price'] <= 0:
                            profit_pct = 0
                        else:
                            profit_pct = ((exit_price - open_position['entry_price']) / open_position['entry_price']) * 100
                        
                        # Create trade record
                        trade = {
                            'watch_id': watch_id,
                            'entry_price': open_position['entry_price'],
                            'entry_time': open_position['entry_time'],
                            'exit_price': exit_price,
                            'exit_time': exit_time,
                            'profit_pct': profit_pct,
                            'trigger_type': trigger,
                            'execution_delay': execution_delay_seconds,
                            'signal_price': open_position['signal_price'], 
                            'signal_time': open_position['signal_time']   
                        }
                        
                        trades.append(trade)
                        trades_per_coin[watch_id] += 1
            # --- End Session End Handling ---

        # Pass the list of all generated trades and parameters to BacktestResult
        # BacktestResult will handle fee calculation, dynamic sizing, and final metrics.
        return BacktestResult(trades, 
                              self.watch_data,
                              initial_capital=initial_capital, 
                              fee_pct=fee_pct,
                              flat_fee_usd=flat_fee_usd, 
                              position_sizing_pct=position_sizing_pct)

# Example usage
if __name__ == "__main__":
    print("Signal-based backtester loaded. Import and use the Backtester and SignalBasedOptimizer classes.")
    print("No default strategies are defined - create your own in strategies.py")
