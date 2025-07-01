"""
Data loading and feature engineering for Reptile trading.

This module provides utilities for fetching market data and
preparing it for meta-learning tasks.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Generator, Optional, List
import requests


def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch historical klines from Bybit.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval in minutes ('1', '5', '15', '60', '240', 'D')
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"API error: {data['retMsg']}")

        df = pd.DataFrame(data['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        df = df.set_index('timestamp').sort_index()

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def create_trading_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Create technical features for trading.

    Args:
        prices: Price series
        window: Lookback window for features

    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=prices.index)

    # Returns at different horizons
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)

    # Moving average ratios
    features['sma_ratio'] = prices / prices.rolling(window).mean() - 1
    features['ema_ratio'] = prices / prices.ewm(span=window).mean() - 1

    # Volatility
    features['volatility'] = prices.pct_change().rolling(window).std()

    # Momentum
    features['momentum'] = prices / prices.shift(window) - 1

    # RSI
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = (100 - (100 / (1 + gain / loss))) / 100 - 0.5  # Normalized

    return features.dropna()


def create_task_data(
    prices: pd.Series,
    features: pd.DataFrame,
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create support and query sets for a trading task.

    Args:
        prices: Price series
        features: Feature DataFrame
        support_size: Number of samples for adaptation
        query_size: Number of samples for evaluation
        target_horizon: Prediction horizon for returns

    Returns:
        (support_data, query_data) tuples
    """
    # Create target (future returns)
    target = prices.pct_change(target_horizon).shift(-target_horizon)

    # Align and drop NaN
    aligned = features.join(target.rename('target')).dropna()

    # Check if we have enough data
    total_needed = support_size + query_size
    if len(aligned) < total_needed:
        raise ValueError(f"Not enough data: {len(aligned)} < {total_needed}")

    # Random split point
    start_idx = np.random.randint(0, len(aligned) - total_needed)

    # Split into support and query
    support_df = aligned.iloc[start_idx:start_idx + support_size]
    query_df = aligned.iloc[start_idx + support_size:start_idx + total_needed]

    # Convert to tensors
    feature_cols = [c for c in aligned.columns if c != 'target']

    support_features = torch.FloatTensor(support_df[feature_cols].values)
    support_labels = torch.FloatTensor(support_df['target'].values).unsqueeze(1)

    query_features = torch.FloatTensor(query_df[feature_cols].values)
    query_labels = torch.FloatTensor(query_df['target'].values).unsqueeze(1)

    return (support_features, support_labels), (query_features, query_labels)


def task_generator(
    asset_data: Dict[str, Tuple[pd.Series, pd.DataFrame]],
    batch_size: int = 4,
    support_size: int = 20,
    query_size: int = 10,
    target_horizon: int = 5
) -> Generator:
    """
    Generate batches of tasks from multiple assets.

    Args:
        asset_data: Dict of {asset_name: (prices, features)}
        batch_size: Number of tasks per batch
        support_size: Samples per support set
        query_size: Samples per query set
        target_horizon: Prediction horizon

    Yields:
        List of tasks
    """
    asset_names = list(asset_data.keys())

    while True:
        tasks = []
        for _ in range(batch_size):
            # Sample random asset
            asset = np.random.choice(asset_names)
            prices, features = asset_data[asset]

            try:
                # Create task data
                support, query = create_task_data(
                    prices, features,
                    support_size=support_size,
                    query_size=query_size,
                    target_horizon=target_horizon
                )
                tasks.append((support, query))
            except ValueError:
                continue

        if tasks:
            yield tasks


def normalize_features(
    features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Normalize features to zero mean and unit variance.

    Args:
        features: Feature DataFrame

    Returns:
        (normalized_features, means, stds)
    """
    means = features.mean()
    stds = features.std().replace(0, 1)  # Avoid division by zero

    normalized = (features - means) / stds

    return normalized, means, stds


def generate_simulated_data(
    symbol: str,
    initial_price: float = 50000.0,
    volatility: float = 0.02,
    num_periods: int = 1000
) -> pd.DataFrame:
    """
    Generate simulated price data for testing.

    Args:
        symbol: Symbol name
        initial_price: Starting price
        volatility: Price volatility (std of returns)
        num_periods: Number of time periods

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    returns = np.random.normal(0, volatility, num_periods)
    prices = initial_price * (1 + returns).cumprod()

    # Generate OHLCV
    df = pd.DataFrame({
        'close': prices,
        'open': np.roll(prices, 1),
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, num_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, num_periods))),
        'volume': np.random.uniform(100, 10000, num_periods),
    })
    df['open'].iloc[0] = initial_price
    df['turnover'] = df['volume'] * df['close']

    # Create timestamp index
    df.index = pd.date_range(end=pd.Timestamp.now(), periods=num_periods, freq='h')
    df.index.name = 'timestamp'

    return df


if __name__ == "__main__":
    # Example usage
    print("Data Loader for Reptile Trading")
    print("=" * 50)

    # Generate simulated data for testing
    print("\nGenerating simulated data...")
    btc_data = generate_simulated_data("BTCUSDT", initial_price=50000.0, num_periods=500)
    eth_data = generate_simulated_data("ETHUSDT", initial_price=3000.0, num_periods=500)

    print(f"BTC data shape: {btc_data.shape}")
    print(f"ETH data shape: {eth_data.shape}")

    # Create features
    print("\nCreating features...")
    btc_features = create_trading_features(btc_data['close'])
    eth_features = create_trading_features(eth_data['close'])

    print(f"BTC features shape: {btc_features.shape}")
    print(f"ETH features shape: {eth_features.shape}")
    print(f"Feature names: {list(btc_features.columns)}")

    # Prepare asset data
    asset_data = {
        'BTCUSDT': (btc_data['close'].loc[btc_features.index], btc_features),
        'ETHUSDT': (eth_data['close'].loc[eth_features.index], eth_features),
    }

    # Create task generator
    print("\nCreating task generator...")
    gen = task_generator(asset_data, batch_size=4)

    # Get a batch of tasks
    tasks = next(gen)
    print(f"Number of tasks in batch: {len(tasks)}")

    for i, (support, query) in enumerate(tasks):
        print(f"\nTask {i+1}:")
        print(f"  Support features: {support[0].shape}")
        print(f"  Support labels: {support[1].shape}")
        print(f"  Query features: {query[0].shape}")
        print(f"  Query labels: {query[1].shape}")

    print("\nDone!")
