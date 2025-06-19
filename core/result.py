import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

class PlottingMixin:
    """Mixin class providing plotting functionality for BacktestResult"""
    
    def plot_results(self, figsize=(15, 12)):
        """Plot comprehensive backtest results based on executed trades"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 2, figure=fig) 
        
        # Plot 1: Net Returns Distribution (% per trade)
        ax1 = fig.add_subplot(gs[0, 0])
        net_returns_pct = []
        for trade in self.executed_trades:
            pos_size = trade.get('position_size_usd', 0)
            if pos_size > 0:
                 net_returns_pct.append((trade.get('net_profit_usd', 0) / pos_size) * 100)

        if net_returns_pct:
             sns.histplot(net_returns_pct, bins=30, ax=ax1, kde=True)
             ax1.set_title(f'Net Returns Distribution ({self.total_executed_trades} Trades)')
             ax1.set_xlabel('Net Return % per Trade')
             ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        else:
             ax1.set_title('No Trades Executed')

        # Plot 2: Cumulative Gross Returns (%)
        ax2 = fig.add_subplot(gs[0, 1])
        # Ensure profit_pct exists and is numeric
        gross_returns = [t['profit_pct'] for t in self.executed_trades if 'profit_pct' in t and pd.notna(t['profit_pct'])]
        if gross_returns:
            cum_returns = np.cumsum(gross_returns)
            # Create an integer index for plotting
            trade_numbers_gross = range(1, len(cum_returns) + 1)
            ax2.plot(trade_numbers_gross, cum_returns)
            ax2.set_title('Cumulative Gross Returns (%)')
            ax2.set_xlabel('Executed Trade Number')
            ax2.set_ylabel('Cumulative Gross Return %')
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
             ax2.set_title('Cumulative Gross Returns (No Valid Data)')

        # Plot 3: Wallet Value in USD
        ax3 = fig.add_subplot(gs[1, :])
        if len(self.wallet_values) > 1:
            # Plotting wallet value over trade number
            trade_numbers_wallet = range(len(self.wallet_values))
            ax3.plot(trade_numbers_wallet, self.wallet_values, color='green', marker='.', markersize=3, linestyle='-')
            ax3.set_title(f'Portfolio Value (USD) | Final: ${self.final_wallet_value:,.2f}')
            ax3.set_xlabel('Trade Number (0 = Initial Capital)')
            ax3.set_ylabel('USD Value')
            ax3.grid(True, alpha=0.5)
            ax3.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label=f'Initial Capital (${self.initial_capital:,.2f})')
            ax3.legend()
            ax3.ticklabel_format(style='plain', axis='y')
            ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax3.set_title('Portfolio Value (USD)')
            ax3.text(0.5, 0.5, 'No trades executed or zero initial capital', ha='center', va='center')

        # Plot 4: Top Coins by Net Profit (USD)
        ax4 = fig.add_subplot(gs[2, 0])
        if self.coin_metrics:
            # Sort by total_net_profit_usd
            coin_net_profits = [(k, v['total_net_profit_usd']) for k, v in self.coin_metrics.items()]
            coin_net_profits.sort(key=lambda x: x[1], reverse=True)
            
            num_coins_to_plot = min(10, len(coin_net_profits)) 
            if num_coins_to_plot > 0:
                coins, net_profits = zip(*coin_net_profits[:num_coins_to_plot])  
                colors = ['g' if p > 0 else 'r' for p in net_profits]
                ax4.bar(range(len(coins)), net_profits, color=colors)
                ax4.set_title('Top Coins by Net Profit (USD)')
                ax4.set_xticks(range(len(coins)))
                ax4.set_xticklabels([c[:8] + '...' for c in coins], rotation=45, ha='right')
                ax4.set_ylabel('Total Net Profit (USD)')
                ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
                ax4.ticklabel_format(style='plain', axis='y')
            else:
                 ax4.set_title('No Coin Data')
        else:
             ax4.set_title('No Coin Data')
        
        # Plot 5: Win Rate
        ax5 = fig.add_subplot(gs[2, 1])
        if self.total_executed_trades > 0:
            labels = ['Winning', 'Losing']
            sizes = [self.winning_trades, self.total_executed_trades - self.winning_trades]
            explode = (0.1, 0)  # explode 1st slice
            ax5.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90, colors=['#8FBC8F', '#FFA07A'])
            ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax5.set_title(f'Win/Loss Ratio (Total: {self.total_executed_trades} Trades)')
        else:
            ax5.set_title('No Executed Trades')
        
        # Plot 6: Trade Duration Distribution
        ax6 = fig.add_subplot(gs[3, :])
        if self.trade_durations:
            sns.histplot(self.trade_durations, kde=True, ax=ax6)
            ax6.set_title('Trade Duration Distribution')
            ax6.set_xlabel('Duration (minutes)')
            ax6.set_ylabel('Count')
        else:
             ax6.set_title('No Trade Duration Data Available')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
        fig.suptitle(f'Backtest Results | Fee: {self.fee_pct*2:.3f}% R/T | Pos Size: {self.position_sizing_pct*100:.1f}% Equity', fontsize=14)
        plt.show()

    def plot_metadata_analysis(self):
        """Plot analysis of metadata relationships with trade performance"""
        if not self.executed_trades:
            print("No executed trades to analyze metadata")
            return
            
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig)
        
        # Create DataFrames for analysis
        trade_df = pd.DataFrame(self.executed_trades)
        trade_df['is_winner'] = trade_df['net_profit_usd'] > 0
        
        # 1. Trade Duration vs Win Rate
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=trade_df, x='is_winner', y='duration', ax=ax1)
        ax1.set_title('Trade Duration by Outcome')
        ax1.set_xlabel('Is Winning Trade')
        ax1.set_ylabel('Duration (minutes)')
        
        # 2. Entry Delay vs Win Rate
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=trade_df, x='is_winner', y='entry_delay', ax=ax2)
        ax2.set_title('Entry Delay by Outcome')
        ax2.set_xlabel('Is Winning Trade')
        ax2.set_ylabel('Entry Delay (minutes)')
        
        # 3. Coin Age vs Win Rate
        ax3 = fig.add_subplot(gs[0, 2])
        sns.boxplot(data=trade_df, x='is_winner', y='age', ax=ax3)
        ax3.set_title('Coin Age by Outcome')
        ax3.set_xlabel('Is Winning Trade')
        ax3.set_ylabel('Age (seconds)')
        
        # 4. Volume vs Win Rate
        ax4 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=trade_df, x='is_winner', y='volume', ax=ax4)
        ax4.set_title('Volume by Outcome')
        ax4.set_xlabel('Is Winning Trade')
        ax4.set_ylabel('Volume (USD)')
        
        # 5. Market Cap vs Win Rate
        ax5 = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=trade_df, x='is_winner', y='mcap', ax=ax5)
        ax5.set_title('Market Cap by Outcome')
        ax5.set_xlabel('Is Winning Trade')
        ax5.set_ylabel('Market Cap (USD)')
        
        # 6. Liquidity vs Win Rate
        ax6 = fig.add_subplot(gs[1, 2])
        sns.boxplot(data=trade_df, x='is_winner', y='liquidity', ax=ax6)
        ax6.set_title('Liquidity by Outcome')
        ax6.set_xlabel('Is Winning Trade')
        ax6.set_ylabel('Liquidity (USD)')
        
        # 7. Entry Timing Distribution
        ax8 = fig.add_subplot(gs[2, 1])
        sns.histplot(data=trade_df, x='entry_delay', hue='is_winner', multiple="layer", ax=ax8)
        ax8.set_title('Entry Timing Distribution')
        ax8.set_xlabel('Entry Delay (minutes)')
        ax8.set_ylabel('Count')
        
        # 8. Trade Duration Distribution
        ax9 = fig.add_subplot(gs[2, 2])
        sns.histplot(data=trade_df, x='duration', hue='is_winner', multiple="layer", ax=ax9)
        ax9.set_title('Trade Duration Distribution')
        ax9.set_xlabel('Duration (minutes)')
        ax9.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical analysis
        print("\n=== Metadata Analysis ===")
        print("\nMedian Values for Winning vs Losing Trades:")
        metrics = ['duration', 'entry_delay', 'age', 'volume', 'mcap', 'liquidity']
        
        medians = trade_df.groupby('is_winner')[metrics].median()
        print("\nWinning Trades Medians:")
        for metric in metrics:
            print(f"{metric:15s}: {medians.loc[True, metric]:10.2f}")
            
        print("\nLosing Trades Medians:")
        for metric in metrics:
            print(f"{metric:15s}: {medians.loc[False, metric]:10.2f}")
            
        # Calculate correlations with profit
        correlations = trade_df[metrics + ['net_profit_usd']].corr()['net_profit_usd'].sort_values(ascending=False)
        print("\nCorrelations with Net Profit:")
        for metric, corr in correlations.items():
            if metric != 'net_profit_usd':
                print(f"{metric:15s}: {corr:10.3f}")

    @staticmethod
    def plot_optimization_results(results_df: pd.DataFrame, best_summary_dict: dict, parameter_ranges: dict):
        """Plot optimization results using Plotly for interactive visualization."""
        # Create a copy to avoid modifying the original
        results_df = results_df.copy()
        best_summary_dict = best_summary_dict.copy() if best_summary_dict else {}

        # Convert percentage strings to float values before plotting
        for col in ['total_return_pct', 'max_drawdown_pct', 'win_rate_pct', 'test_return_pct', 
                   'test_drawdown_pct', 'test_win_rate_pct', 'score', 'trades']:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col].astype(str).str.rstrip('%').str.rstrip('f'), errors='coerce')
            if col in best_summary_dict:
                try:
                    val = str(best_summary_dict[col]).rstrip('%').rstrip('f')
                    best_summary_dict[col] = float(val)
                except (ValueError, AttributeError):
                    try:
                        best_summary_dict[col] = float(best_summary_dict[col])
                    except (ValueError, TypeError):
                        best_summary_dict[col] = 0.0

        # Check if walk-forward optimization was used (presence of test metrics indicates WFO)
        has_wfo = best_summary_dict and all(k in best_summary_dict for k in ['test_return_pct', 'test_drawdown_pct', 'test_win_rate_pct'])
        
        # Create subplot figure based on whether WFO was used
        if has_wfo:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    'Returns vs Drawdown (All Parameter Combinations)',
                    'Training vs Test Performance'
                ),
                specs=[[{"type": "scatter"}, {"type": "bar"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=('Returns vs Drawdown (All Parameter Combinations)',)
            )

        # 1. Returns vs Drawdown Scatter Plot
        param_cols = [col for col in results_df.columns if col in parameter_ranges.keys()]
        
        # Create hover text with safe value conversion
        hover_text = []
        for _, row in results_df.iterrows():
            try:
                param_texts = []
                for param in param_cols:
                    try:
                        val = float(row[param])
                        param_texts.append(f"{param}: {val:.2f}")
                    except (ValueError, TypeError):
                        param_texts.append(f"{param}: N/A")
                
                metric_texts = [
                    "<br><b>Metrics:</b>",
                    f"Return: {float(row['total_return_pct']):.2f}%",
                    f"Drawdown: {float(row['max_drawdown_pct']):.2f}%",
                    f"Win Rate: {float(row['win_rate_pct']):.2f}%",
                    f"Trades: {int(float(row['trades']))}",
                    f"Score: {float(row['score']):.2f}"
                ]
                
                # Add optional metrics if they exist
                if 'density_score' in row:
                    try:
                        metric_texts.append(f"Density: {float(row['density_score']):.3f}")
                    except (ValueError, TypeError):
                        metric_texts.append("Density: N/A")
                        
                if 'trade_factor' in row:
                    try:
                        metric_texts.append(f"Trade Factor: {float(row['trade_factor']):.3f}")
                    except (ValueError, TypeError):
                        metric_texts.append("Trade Factor: N/A")
                        
                if 'is_outlier' in row:
                    metric_texts.append(f"Is Outlier: {row['is_outlier']}")
                
                text = "<br>".join(["<b>Parameters:</b>"] + param_texts + metric_texts)
                hover_text.append(text)
            except Exception as e:
                hover_text.append("Error processing data point")

        # Add scatter plot for all points
        if has_wfo:
            col_position = 1
        else:
            col_position = 1  # Single column layout
            
        fig.add_trace(
            go.Scatter(
                x=results_df['max_drawdown_pct'],
                y=results_df['total_return_pct'],
                mode='markers',
                name='Results',
                marker=dict(
                    size=8,
                    color=results_df['score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                text=hover_text,
                hoverinfo='text',
            ),
            row=1, col=col_position
        )

        # Add best point if available
        if best_summary_dict:
            try:
                param_texts = []
                for param in param_cols:
                    try:
                        val = float(best_summary_dict.get(param, 0))
                        param_texts.append(f"{param}: {val:.2f}")
                    except (ValueError, TypeError):
                        param_texts.append(f"{param}: N/A")
                
                best_hover = "<br>".join(
                    ["<b>BEST PARAMETERS:</b>"] +
                    param_texts +
                    ["<br><b>Training Metrics:</b>",
                    f"Return: {float(best_summary_dict['total_return_pct']):.2f}%",
                    f"Drawdown: {float(best_summary_dict['max_drawdown_pct']):.2f}%",
                    f"Win Rate: {float(best_summary_dict['win_rate_pct']):.2f}%",
                    f"Trades: {int(float(best_summary_dict['trades']))}",
                    f"Score: {float(best_summary_dict['score']):.2f}"]
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[float(best_summary_dict['max_drawdown_pct'])],
                        y=[float(best_summary_dict['total_return_pct'])],
                        mode='markers',
                        name='Best Parameters',
                        marker=dict(
                            size=15,
                            symbol='star',
                            color='red'
                        ),
                        text=best_hover,
                        hoverinfo='text'
                    ),
                    row=1, col=col_position
                )
            except Exception as e:
                print(f"Warning: Could not add best point to plot: {e}")

        # 2. Training vs Test Performance Comparison (only if WFO is enabled)
        if has_wfo:
            try:
                metrics = ['Return', 'Drawdown', 'Win Rate']
                train_values = [
                    float(best_summary_dict['total_return_pct']),
                    float(best_summary_dict['max_drawdown_pct']),
                    float(best_summary_dict['win_rate_pct'])
                ]
                test_values = [
                    float(best_summary_dict['test_return_pct']),
                    float(best_summary_dict['test_drawdown_pct']),
                    float(best_summary_dict['test_win_rate_pct'])
                ]

                fig.add_trace(
                    go.Bar(
                        x=metrics,
                        y=train_values,
                        name='Training'
                    ),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Bar(
                        x=metrics,
                        y=test_values,
                        name='Test'
                    ),
                    row=1, col=2
                )
            except Exception as e:
                print(f"Warning: Could not add training vs test comparison: {e}")

        # Update layout
        if has_wfo:
            title_text = "Optimization Results (Walk-Forward Analysis)"
            height = 600
        else:
            title_text = "Optimization Results"
            height = 600
            
        fig.update_layout(
            title_text=title_text,
            height=height,
            showlegend=True,
            hovermode='closest'
        )

        # Save the plot as HTML and open in browser
        try:
            output_path = os.path.join(os.getcwd(), 'optimization_results.html')
            fig.write_html(output_path)
            webbrowser.open('file://' + output_path)
        except Exception as e:
            print(f"Warning: Could not save or open plot: {e}")

        # Print the text summary
        try:
            PlottingMixin._print_optimization_summary(results_df, best_summary_dict, parameter_ranges)
        except Exception as e:
            print(f"Warning: Could not print optimization summary: {e}")

    @staticmethod
    def _print_optimization_summary(results_df: pd.DataFrame, best_summary_dict: dict, parameter_ranges: dict):
        """Print detailed optimization results summary."""
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*50)
        
        if best_summary_dict:
            param_cols = [col for col in results_df.columns if col in parameter_ranges.keys()]
            print("\nBest Parameters:")
            for param in param_cols:
                print(f"{param:20s}: {best_summary_dict[param]:>10.2f}")
            
            print("\nTraining Set Performance:")
            print(f"{'Total Return':20s}: {best_summary_dict['total_return_pct']:>10.2f}%")
            print(f"{'Max Drawdown':20s}: {best_summary_dict['max_drawdown_pct']:>10.2f}%")
            print(f"{'Win Rate':20s}: {best_summary_dict['win_rate_pct']:>10.2f}%")
            print(f"{'Number of Trades':20s}: {int(float(best_summary_dict['trades'])):>10d}")
            print(f"{'Score':20s}: {float(best_summary_dict['score']):>10.2f}")
            
            if 'test_return_pct' in best_summary_dict:
                print("\nTest Set Performance:")
                print(f"{'Total Return':20s}: {best_summary_dict['test_return_pct']:>10.2f}%")
                print(f"{'Max Drawdown':20s}: {best_summary_dict['test_drawdown_pct']:>10.2f}%")
                print(f"{'Win Rate':20s}: {best_summary_dict['test_win_rate_pct']:>10.2f}%")
                print(f"{'Number of Trades':20s}: {int(float(best_summary_dict['test_trades'])):>10d}")
                
                # Calculate performance difference
                print("\nTraining vs Test Difference:")
                return_diff = best_summary_dict['test_return_pct'] - best_summary_dict['total_return_pct']
                drawdown_diff = best_summary_dict['test_drawdown_pct'] - best_summary_dict['max_drawdown_pct']
                winrate_diff = best_summary_dict['test_win_rate_pct'] - best_summary_dict['win_rate_pct']
                
                print(f"{'Return Delta':20s}: {return_diff:>10.2f}%")
                print(f"{'Drawdown Delta':20s}: {drawdown_diff:>10.2f}%")
                print(f"{'Win Rate Delta':20s}: {winrate_diff:>10.2f}%")
        
        print("="*50)
