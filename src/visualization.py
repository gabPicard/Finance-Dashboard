import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def portfolio_analysis(data, title, prices=None, realized_returns=None, type='Full'):
    """
    Visualize portfolio optimization results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Weights backtest data from rolling_window
    title : str
        Title for the figure
    prices : pd.DataFrame, optional
        Price data to calculate realized returns
    realized_returns : dict, optional
        Pre-calculated realized returns (from calculate_realized_returns)
    type : str
        Type of visualization ('Full', 'portfolio only', 'metrics only')
    """
    if type == "Full":
        # Create larger figure with space for stats
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.4)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Define axes
        ax1 = fig.add_subplot(gs[0, :2])  # Asset weights
        ax2 = fig.add_subplot(gs[1, 0])   # Sharpe ratio
        ax3 = fig.add_subplot(gs[1, 1])   # Expected return
        ax4 = fig.add_subplot(gs[2, 0])   # Standard deviation
        ax5 = fig.add_subplot(gs[2, 1])   # Portfolio value
        ax_stats = fig.add_subplot(gs[:, 2])  # Stats panel

        # Asset weights over time
        asset_columns = [col for col in data.columns if col not in ['sharpe_ratio', 'expected_return', 'std']]
        data[asset_columns].abs().plot(kind='area', stacked=True, cmap='tab20', ax=ax1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weight')
        ax1.set_title('Asset Weights Over Time', fontweight='bold')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Sharpe ratio over time
        ax2.plot(data.index, data['sharpe_ratio'], color='green', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio Over Time')
        ax2.grid(True, alpha=0.3)

        # Expected return over time
        ax3.plot(data.index, data['expected_return'], color='blue', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Expected Return')
        ax3.set_title('Expected Return Over Time')
        ax3.grid(True, alpha=0.3)

        # Standard deviation over time
        ax4.plot(data.index, data['std'], color='red', linewidth=2)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Standard Deviation Over Time')
        ax4.grid(True, alpha=0.3)
        
        # Portfolio value over time and statistics
        if realized_returns is not None:
            portfolio_value = realized_returns['portfolio_value']
            cumulative_return = realized_returns['cumulative_return']
            annualized_return = realized_returns['annualized_return']
            avg_sharpe = realized_returns['avg_sharpe']
            avg_std = realized_returns['avg_std']
            final_value = realized_returns['final_value']
            initial_value = realized_returns['initial_value']
            
            # Plot portfolio value
            ax5.plot(portfolio_value.index, portfolio_value.values, color='purple', linewidth=2.5)
            ax5.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label=f'Initial: ${initial_value:.2f}')
            ax5.fill_between(portfolio_value.index, initial_value, portfolio_value.values, 
                            where=(portfolio_value.values >= initial_value), 
                            alpha=0.3, color='green', interpolate=True)
            ax5.fill_between(portfolio_value.index, initial_value, portfolio_value.values,
                            where=(portfolio_value.values < initial_value),
                            alpha=0.3, color='red', interpolate=True)
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Portfolio Value ($)')
            ax5.set_title('Portfolio Value Over Time')
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)
            
            # Statistics panel
            ax_stats.axis('off')
            
            # Create nice stats box
            stats_text = f"""
╔══════════════════════════════════╗
║     PORTFOLIO SUMMARY STATS      ║
╚══════════════════════════════════╝

📊 PERFORMANCE METRICS
{'─' * 38}
Initial Investment:      ${initial_value:>10.2f}
Final Value:            ${final_value:>10.2f}
Total Return:           {cumulative_return:>10.2%}
Annualized Return:      {annualized_return:>10.2%}

📈 RISK METRICS
{'─' * 38}
Average Sharpe Ratio:   {avg_sharpe:>10.3f}
Average Std Dev:        {avg_std:>10.2%}

⏱️ PERIOD
{'─' * 38}
Start Date:   {portfolio_value.index[0].strftime('%Y-%m-%d')}
End Date:     {portfolio_value.index[-1].strftime('%Y-%m-%d')}
Duration:     {(portfolio_value.index[-1] - portfolio_value.index[0]).days} days

🎯 INTERPRETATION
{'─' * 38}
"""
            
            # Add interpretation
            if cumulative_return > 0:
                profit = final_value - initial_value
                stats_text += f"✓ Portfolio gained ${profit:.2f}\n"
                stats_text += f"  ({(cumulative_return * 100):.2f}% increase)\n\n"
            else:
                loss = initial_value - final_value
                stats_text += f"✗ Portfolio lost ${loss:.2f}\n"
                stats_text += f"  ({(abs(cumulative_return) * 100):.2f}% decrease)\n\n"
            
            if annualized_return > 0.05:
                stats_text += "✓ Strong annualized returns\n"
            elif annualized_return > 0:
                stats_text += "○ Moderate annualized returns\n"
            else:
                stats_text += "✗ Negative annualized returns\n"
            
            if avg_sharpe > 1:
                stats_text += "✓ Excellent risk-adjusted returns\n"
            elif avg_sharpe > 0.5:
                stats_text += "○ Good risk-adjusted returns\n"
            else:
                stats_text += "✗ Poor risk-adjusted returns\n"
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        else:
            # No realized returns available
            ax5.text(0.5, 0.5, 'No price data provided\nfor portfolio value calculation',
                    ha='center', va='center', fontsize=12, color='gray')
            ax5.set_title('Portfolio Value Over Time')
            ax_stats.axis('off')
            ax_stats.text(0.5, 0.5, 'Provide prices data to\ncalculate realized returns',
                         ha='center', va='center', fontsize=12, color='gray',
                         transform=ax_stats.transAxes)

    elif type == "portfolio only":
        ...
    elif type == "metrics only":
        ...

    plt.show()