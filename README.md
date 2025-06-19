# Algorithmic Trading Backtester

This project is a high-performance, feature-rich backtesting engine designed for algorithmic trading strategies on cryptocurrency data. It allows for rigorous testing, optimization, and analysis of trading strategies before deployment. The system is built with Python and leverages powerful libraries like Dask for parallel computing, Numba for speed, and Plotly for interactive visualizations.

## Key Features

### 1. Advanced Backtesting Engine
The core of the project is a sophisticated backtester that simulates trading with a high degree of realism.

- **Realistic Cost Simulation**: Models both percentage-based (`fee_pct`) and flat (`flat_fee_usd`) trading fees for both entry and exit of a trade.
- **Dynamic Position Sizing**: Allocates a specified percentage of the current portfolio equity (`position_sizing_pct`) to each new trade, allowing the strategy to compound returns.
- **Execution Delay (Slippage)**: Simulates the delay between a signal firing and the trade executing (`execution_delay_seconds`), providing a more accurate execution price.
- **Flexible Risk Management**:
    - **Stop Loss**: Apply a global stop-loss percentage (`stop_loss_pct`) to all trades.
    - **Max Trade Duration**: Automatically exit trades that run longer than a specified number of minutes.
- **Signal & Entry Control**:
    - **Max Trades Per Coin**: Limit the number of trades a strategy can take on a single asset.
    - **Max Watch for Entry**: Define a window of time after a session starts within which an entry signal must be found.

### 2. Powerful Parameter Optimizer
The optimizer is designed to find the best parameters for your strategy efficiently using parallel processing.

- **Parallel Processing with Dask**: Distributes backtesting tasks across all available CPU cores, dramatically speeding up optimization.
- **Walk-Forward Optimization (WFO)**: Mitigate overfitting by testing parameters on a rolling forward basis. The dataset is split into training and testing sets chronologically (`use_walk_forward`, `train_size`).
- **Customizable Scoring Logic**: The "best" parameter set is determined by a scoring function. The current implementation rewards a high **return-to-drawdown ratio**, but this can be easily customized in `core/optimizer.py`.
- **Flexible Parameter Ranges**: Define start, end, and step values for any strategy or backtester parameter you wish to optimize.
- **Random Sampling**: For very large parameter spaces, you can test a random subset of combinations (`random_sample_size`) to get a good approximation of the best parameters faster.

### 3. Data & Strategy Integrity Tools
Several features are included to ensure your strategy is robust and not just curve-fit to the data sequence.

- **Scramble Mode**: Randomizes the order of the trading sessions (`watch_tracking.csv`) before running a backtest. This helps verify that the strategy's performance is not dependent on the chronological sequence of events.
- **Start Line Capability**: Begin a backtest from any point within the dataset (`start_line`), useful for debugging or focusing on specific market periods.
- **Modular Strategies**: Strategies are self-contained classes, making it easy to add new ones without altering the core engine. Example included:
    - `SMAStrategy` (Simple Moving Average Crossover)

### 4. Comprehensive Analysis & Visualization
The backtester provides rich, multi-faceted insights into a strategy's performance.

- **Detailed Console Reports**: Get a full summary of performance, including net return, max drawdown, win rate, trade duration analysis, and fee breakdowns.
- **Standard Backtest Plots**: The `plot_results()` function generates a dashboard of Matplotlib charts showing the equity curve, returns distribution, win/loss pie chart, and more.
- **Interactive Optimization Plots**: The optimizer generates an interactive HTML file (`optimization_results.html`) using Plotly, allowing you to explore the relationship between parameters, returns, drawdowns, and the final score.
- **Metadata Analysis**: Uncover correlations between trade performance and coin characteristics like market cap, liquidity, volume, and age.

### 5. Advanced Data Management & Generation
The system includes sophisticated data handling capabilities for both real and synthetic data.

- **Dual Data Structure**: 
  - `price_history.csv`: High-frequency tick-level price data with timestamps
  - `watch_tracking.csv`: Session-level metadata including market cap, volume, liquidity, and coin age
- **Synthetic Data Generation**: Built-in tools to generate realistic cryptocurrency price data for testing and development
- **Performance Optimization**: Automatic conversion to Parquet format for faster I/O operations
- **Data Analysis Tools**: Comprehensive analyzer for exploring data characteristics and quality
- **Flexible Data Loading**: On-demand loading with intelligent caching for memory efficiency

## Project Structure

```
demo_backtester/
│
├── core/                   # Core engine components
│   ├── backtester.py       # The main backtesting engine
│   ├── optimizer.py        # Parameter optimization logic
│   ├── result.py           # Result calculation and plotting
│   ├── strategies.py       # Container for abstract strategy template
│   └── sma_strategy.py     # Example of a standalone strategy file
│
├── data/                   # Data files and data-related scripts
│   ├── price_history.csv   # Tick-level price data (timestamp, price, watch_id)
│   ├── watch_tracking.csv  # Session metadata (watch_id, start/end times, mcap, volume, etc.)
│   ├── analyzer.py         # Data analysis and exploration tools
│   ├── generate_bogus_data.py  # Synthetic data generation for testing
│   └── convert_to_parquet.py   # Performance optimization utility
│
├── run_sma_backtest.py       # Example script to run the SMA strategy
├── requirements.txt          # Project dependencies
└── readme.md                 # This file
```

## Installation

1.  Clone the repository to your local machine.
2.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3.  **Convert data to Parquet format** (Required before first use):
    ```sh
    python data/convert_to_parquet.py
    ```
    This step optimizes data loading performance and is required for the backtester to function properly.

## How to Run

You can run a backtest or an optimization by executing one ofthe run scripts, such as `run_sma_backtest.py`.

```sh
python run_sma_backtest.py
```

Inside the script, you can easily switch between modes:

-   **Basic Backtest**: Calls `run_basic_backtest()` to test the strategy with a single, fixed set of parameters.
-   **Parameter Optimization**: Calls `run_parameter_optimization()` to launch the Dask-powered optimizer and find the best parameters based on the defined ranges and scoring logic.

## Data Management

### Data Structure
The backtester uses a dual-file data structure optimized for both performance and flexibility:

**`price_history.csv`** - Contains tick-level price data:
- `timestamp`: Datetime of each price tick
- `price`: Asset price at that timestamp  
- `watch_id`: Unique identifier linking to session metadata

**`watch_tracking.csv`** - Contains session-level metadata:
- `watch_id`: Unique session identifier
- `watch_start_time` / `watch_end_time`: Session duration
- `mcap`: Market capitalization
- `vol`: Trading volume
- `liq`: Liquidity metrics
- `age`: Asset age in days

### Data Generation & Testing
The system includes powerful tools for generating synthetic data for testing and development:

**`generate_bogus_data.py`** - Creates realistic synthetic cryptocurrency data:
- Generates both price data and metadata
- Simulates realistic market conditions and volatility
- Creates correlated data relationships (volume, market cap, etc.)
- Useful for strategy development and testing without real market data

**`analyzer.py`** - Comprehensive data analysis toolkit:
- Data quality assessment and validation
- Statistical analysis of price movements and metadata
- Correlation analysis between different data features
- Data distribution visualization and summary statistics
- Identifies potential data issues or anomalies

**`convert_to_parquet.py`** - Performance optimization utility:
- Converts CSV files to Parquet format for faster I/O
- Significantly improves backtesting speed with large datasets
- Maintains data integrity while reducing file size
- **Required to run before using the backtester** - the system expects Parquet files for optimal performance

### Usage Examples

**Generate synthetic data for testing:**
```python
python data/generate_bogus_data.py
```

**Analyze your data:**
```python
python data/analyzer.py
```

**Optimize data performance (Required before first backtest):**
```python
python data/convert_to_parquet.py
```

## How to Add a New Strategy

1.  **Create a Strategy File**: Create a new Python file in the `core/` directory (e.g., `my_strategy.py`).
2.  **Define Parameters**: Use a `@dataclass` to define the parameters for your strategy, inheriting from `StrategyParams`.
3.  **Implement the Strategy Class**:
    -   Create a class that inherits from the `Strategy` abstract base class.
    -   Implement the `get_name()` and `get_description()` methods.
    -   Implement the `generate_signals()` method. This is the core of your strategy, where you analyze the price data (`pd.DataFrame`) and return a list of buy/sell signals as `(SignalType, price, timestamp)` tuples.
    -   For performance, it is highly recommended to write calculation-heavy logic in a Numba-jitted function (`@numba.jit`).
4.  **Create a Run Script**: Copy an existing script like `run_sma_backtest.py` and modify it to import and use your new strategy.
