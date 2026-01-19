import json
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.Markowitz import calculate_efficient_frontier, best_sharpe_ratio
from data.fetch_data import fetch_stock_data, fetch_risk_free_rate


def read_json(filename: str) -> Dict:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{filename}': {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {e}")
        return None


def write_json(filename: str, data: Dict) -> bool:
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error writing to '{filename}': {e}")
        return False


def run_markowitz_optimization(tickers: List[str], filename) -> Dict:
    data = read_json(filename)
    data =  data['Markowitz Model']

    param = {
        "short selling": data['Possible parameters']['short selling'],
        "calculation window": data['Possible parameters']['calculation window'],
        "period": data['Possible parameters']['period']
    }




