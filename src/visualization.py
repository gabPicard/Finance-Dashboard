import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def portfolio_analysis(data, title, type='Full'):
    if type == "Full":
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        # Asset weights over time
        ax1 = axes[0, 0]
        asset_columns = [col for col in data.columns if col not in ['sharpe_ratio', 'expected_return', 'std']]
        data[asset_columns].abs().plot(kind='area', stacked=True, cmap='tab20', ax=ax1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weight')
        ax1.set_title('Asset Weights Over Time')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)

        # Sharpe ratio over time
        ax2 = axes[0, 1]
        ax2.plot(data.index, data['sharpe_ratio'], color='green', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio Over Time')
        ax2.grid(True, alpha=0.3)

        # Expected return over time
        ax3 = axes[1, 0]
        ax3.plot(data.index, data['expected_return'], color='blue', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Expected Return')
        ax3.set_title('Expected Return Over Time')
        ax3.grid(True, alpha=0.3)

        # Standard deviation over time
        ax4 = axes[1, 1]
        ax4.plot(data.index, data['std'], color='red', linewidth=2)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Standard Deviation Over Time')
        ax4.grid(True, alpha=0.3)

    elif type == "portfolio only":
        ...
    elif type == "metrics only":
        ...

    plt.show()