from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import numba

# --- Base Classes and Enums (included for modularity) ---
class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class StrategyParams:
    """Base class for strategy parameters."""
    pass

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> List[Tuple[str, float, pd.Timestamp]]:
        """Generate signals from price data."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for the strategy."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of how the strategy works."""
        pass

# --- Numba Optimized Helper Functions ---
@numba.jit(nopython=True)
def calculate_sma_nb(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using Numba for performance.
    
    Args:
        prices: Numpy array of prices.
        window: The moving average window period.
        
    Returns:
        Numpy array of SMA values.
    """
    n = len(prices)
    sma = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return sma
    
    # Calculate initial sum for the first window
    current_sum = 0.0
    for i in range(window):
        current_sum += prices[i]
    
    sma[window - 1] = current_sum / window
    
    # Use a sliding window for efficiency
    for i in range(window, n):
        current_sum += prices[i] - prices[i - window]
        sma[i] = current_sum / window
        
    return sma

@numba.jit(nopython=True)
def _generate_sma_signals_nb(
    price_arr: np.ndarray,
    short_sma: np.ndarray,
    long_sma: np.ndarray,
    long_window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized function to find SMA crossover signals.
    Returns indices, types (1 for buy, -1 for sell), and prices of signals.
    """
    n = len(price_arr)
    # Pre-allocate arrays for signals. Maximum possible is n.
    signal_indices = np.empty(n, dtype=np.int64)
    signal_types = np.empty(n, dtype=np.int8)
    signal_prices = np.empty(n, dtype=np.float64)
    signal_count = 0
    
    in_position = False
    
    # Start from where the long_sma is valid
    for i in range(long_window, n):
        # Skip if any value is NaN
        if np.isnan(short_sma[i-1]) or np.isnan(short_sma[i]) or \
           np.isnan(long_sma[i-1]) or np.isnan(long_sma[i]):
            continue
            
        # Buy Signal (Golden Cross)
        if not in_position and short_sma[i-1] <= long_sma[i-1] and short_sma[i] > long_sma[i]:
            signal_indices[signal_count] = i
            signal_types[signal_count] = 1 # Buy
            signal_prices[signal_count] = price_arr[i]
            signal_count += 1
            in_position = True
            
        # Sell Signal (Death Cross)
        elif in_position and short_sma[i-1] >= long_sma[i-1] and short_sma[i] < long_sma[i]:
            signal_indices[signal_count] = i
            signal_types[signal_count] = -1 # Sell
            signal_prices[signal_count] = price_arr[i]
            signal_count += 1
            in_position = False
            
    return (
        signal_indices[:signal_count],
        signal_types[:signal_count],
        signal_prices[:signal_count]
    )

# --- SMA Strategy ---
@dataclass
class SMAParams(StrategyParams):
    """Parameters for the SMA Crossover strategy."""
    short_window: int = 20
    long_window: int = 50

class SMAStrategy(Strategy):
    """
    A Simple Moving Average (SMA) Crossover Strategy.
    - Buys when the short-term SMA crosses above the long-term SMA (Golden Cross).
    - Sells when the short-term SMA crosses below the long-term SMA (Death Cross).
    """
    def __init__(self, params: SMAParams = None):
        self.params = params or SMAParams()
        # Ensure window parameters are integers
        self.params.short_window = int(self.params.short_window)
        self.params.long_window = int(self.params.long_window)
        if self.params.short_window >= self.params.long_window:
            raise ValueError("The short window must be smaller than the long window for SMA crossover.")

    def get_name(self) -> str:
        return f"SMA Crossover ({self.params.short_window}/{self.params.long_window})"

    def get_description(self) -> str:
        return (f"Buys when {self.params.short_window}-period SMA crosses above "
                f"{self.params.long_window}-period SMA. Sells on the reverse cross.")

    def generate_signals(self, prices: pd.DataFrame) -> List[Tuple[str, float, pd.Timestamp]]:
        """
        Generates buy/sell signals based on SMA crossovers using a Numba-optimized loop.
        """
        # Ensure there's enough data for the long window
        if len(prices) < self.params.long_window:
            return []

        price_arr = prices['price'].values
        time_arr = prices['timestamp'].values
        
        # Calculate SMAs using the Numba-optimized function
        short_sma = calculate_sma_nb(price_arr, self.params.short_window)
        long_sma = calculate_sma_nb(price_arr, self.params.long_window)
        
        # Call the Numba helper to find signal indices, types, and prices
        signal_indices, signal_types, signal_prices = _generate_sma_signals_nb(
            price_arr,
            short_sma,
            long_sma,
            self.params.long_window
        )
        
        # If no signals, return early
        if len(signal_indices) == 0:
            return []

        # Convert results from Numba into the required list of tuples format
        buy_str = SignalType.BUY.value
        sell_str = SignalType.SELL.value
        signal_timestamps = pd.to_datetime(time_arr[signal_indices])

        signals = [
            (buy_str if signal_types[i] == 1 else sell_str, signal_prices[i], signal_timestamps[i])
            for i in range(len(signal_indices))
        ]
        
        return signals 