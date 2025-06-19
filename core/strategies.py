from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class StrategyParams:
    """Base class for strategy parameters"""
    pass

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> List[Tuple[str, float, pd.Timestamp]]:
        """
        Generate buy/sell signals from price data
        
        Args:
            prices: DataFrame with columns ['timestamp', 'price']
        
        Returns:
            List of tuples (signal_type, price, timestamp) where:
              - signal_type is 'buy' or 'sell'
              - price is the price at which the signal is generated
              - timestamp is the time when the signal is generated
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for the strategy"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of how the strategy works"""
        pass
