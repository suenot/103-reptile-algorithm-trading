# Chapter 82: Reptile Algorithm for Algorithmic Trading

## Overview

The Reptile algorithm is a simple and scalable meta-learning method developed by OpenAI that enables models to quickly adapt to new tasks with minimal data. Unlike its more complex cousin MAML (Model-Agnostic Meta-Learning), Reptile achieves comparable performance while being computationally more efficient and easier to implement.

In the context of algorithmic trading, Reptile is particularly valuable for adapting trading strategies to new market conditions, different assets, or changing market regimes with only a few examples of the new environment.

## Table of Contents

1. [Introduction to Reptile](#introduction-to-reptile)
2. [How Reptile Differs from MAML](#how-reptile-differs-from-maml)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Reptile for Trading Applications](#reptile-for-trading-applications)
5. [Implementation in Python](#implementation-in-python)
6. [Implementation in Rust](#implementation-in-rust)
7. [Practical Examples with Stock and Crypto Data](#practical-examples-with-stock-and-crypto-data)
8. [Backtesting Framework](#backtesting-framework)
9. [Performance Evaluation](#performance-evaluation)
10. [Future Directions](#future-directions)

---

## Introduction to Reptile

### What is Meta-Learning?

Meta-learning, or "learning to learn," is a paradigm where a model learns not just to perform a specific task, but to quickly adapt to new tasks. Think of it as teaching someone how to learn effectively rather than teaching them specific facts.

### The Reptile Algorithm

Reptile was introduced by Nichol et al. (2018) as a first-order meta-learning algorithm. The key insight is remarkably simple:

1. Sample a task from a distribution of tasks
2. Train the model on that task for several steps
3. Move the initialization parameters towards the trained parameters
4. Repeat

The name "Reptile" comes from the fact that the algorithm "slithers" towards good initializations by taking small steps in the direction of task-specific optimal parameters.

### Why Reptile for Trading?

Financial markets exhibit several characteristics that make meta-learning attractive:

- **Regime Changes**: Markets transition between bull, bear, and sideways regimes
- **Cross-Asset Transfer**: Patterns learned on one asset may transfer to others
- **Limited Data in New Conditions**: When market conditions change, historical data may not be representative
- **Fast Adaptation Required**: Markets move quickly, requiring rapid strategy adjustment

---

## How Reptile Differs from MAML

### MAML (Model-Agnostic Meta-Learning)

MAML optimizes for the initialization that, after one or few gradient steps, yields good performance across tasks. It requires computing second-order derivatives (gradients of gradients), which is computationally expensive.

```
MAML: θ ← θ - α ∇_θ Σ_τ L(f_{θ'_τ}(x), y)
where θ'_τ = θ - β ∇_θ L(f_θ(x), y)  [task-specific update]
```

### Reptile

Reptile achieves similar goals but uses only first-order derivatives:

```
Reptile: θ ← θ + ε (θ̃ - θ)
where θ̃ = SGD(θ, τ, k)  [k steps of SGD on task τ]
```

### Key Differences

| Aspect | MAML | Reptile |
|--------|------|---------|
| Gradient Order | Second-order | First-order |
| Computational Cost | High | Low |
| Memory Requirement | High (computation graph) | Low |
| Implementation Complexity | Complex | Simple |
| Performance | Excellent | Comparable |

---

## Mathematical Foundation

### The Reptile Update Rule

Given:
- θ: Current initialization parameters
- τ: A task sampled from task distribution p(τ)
- k: Number of SGD steps on the task
- ε: Meta-learning rate (step size)

The Reptile update is:

```
θ ← θ + ε (θ̃_k - θ)
```

Where θ̃_k represents the parameters after k steps of SGD on task τ starting from θ.

### Understanding Why Reptile Works

Reptile performs implicit gradient descent on the expected loss across tasks. The update direction (θ̃_k - θ) contains information about:

1. **Task-specific gradients**: The direction that improves performance on task τ
2. **Curvature information**: Higher-order gradient information accumulated over k steps

The key insight from the Reptile paper is that:

```
E[θ̃_k - θ] ≈ E[g₁] + O(k²) terms involving Hessians
```

Where g₁ is the gradient at θ, and the higher-order terms help find a good initialization for fast adaptation.

### Batched Reptile

For improved stability, we can average over multiple tasks per update:

```
θ ← θ + ε * (1/n) Σᵢ (θ̃ᵢ - θ)
```

This reduces variance in the meta-learning update direction.

---

## Reptile for Trading Applications

### 1. Multi-Asset Adaptation

Train on multiple assets simultaneously to learn a model that can quickly adapt to any asset:

```
Tasks = {Stock_A, Stock_B, Crypto_X, Crypto_Y, ...}
Each task: Predict next-period return given historical features
```

### 2. Regime Adaptation

Define tasks based on market regimes:

```
Tasks = {Bull_Market_Data, Bear_Market_Data, High_Volatility_Data, Low_Volatility_Data}
Goal: Quick adaptation when regime changes are detected
```

### 3. Time-Period Adaptation

Sample tasks from different time periods:

```
Tasks = {Jan-Mar_2023, Apr-Jun_2023, Jul-Sep_2023, ...}
Goal: Learn patterns that generalize across different market conditions
```

### 4. Strategy Adaptation

Meta-learn across different trading strategy types:

```
Tasks = {Momentum_Strategy, Mean_Reversion_Strategy, Breakout_Strategy}
Goal: Initialize a model that can quickly specialize to any strategy type
```

---

## Implementation in Python

### Core Reptile Algorithm

```python
import torch
import torch.nn as nn
from typing import List, Tuple
import copy

class ReptileTrader:
    """
    Reptile meta-learning algorithm for trading strategy adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5
    ):
        """
        Initialize Reptile trader.

        Args:
            model: Neural network model for trading predictions
            inner_lr: Learning rate for task-specific adaptation
            outer_lr: Meta-learning rate (epsilon in Reptile paper)
            inner_steps: Number of SGD steps per task (k in Reptile paper)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def inner_loop(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[nn.Module, float]:
        """
        Perform task-specific adaptation (inner loop).

        Args:
            support_data: (features, labels) for adaptation
            query_data: (features, labels) for evaluation

        Returns:
            Adapted model and query loss
        """
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        features, labels = support_data

        # Perform k steps of SGD on the task
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            inner_optimizer.step()

        # Evaluate on query set
        with torch.no_grad():
            query_features, query_labels = query_data
            query_predictions = adapted_model(query_features)
            query_loss = nn.MSELoss()(query_predictions, query_labels).item()

        return adapted_model, query_loss

    def meta_train_step(
        self,
        tasks: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                          Tuple[torch.Tensor, torch.Tensor]]]
    ) -> float:
        """
        Perform one meta-training step using Reptile.

        Args:
            tasks: List of (support_data, query_data) tuples

        Returns:
            Average query loss across tasks
        """
        # Store original parameters
        original_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }

        # Accumulate parameter updates across tasks
        param_updates = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }

        total_query_loss = 0.0

        for support_data, query_data in tasks:
            # Reset to original parameters
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(original_params[name])

            # Perform inner loop adaptation
            adapted_model, query_loss = self.inner_loop(support_data, query_data)
            total_query_loss += query_loss

            # Compute parameter difference (θ̃ - θ)
            with torch.no_grad():
                for (name, param), (_, adapted_param) in zip(
                    self.model.named_parameters(),
                    adapted_model.named_parameters()
                ):
                    param_updates[name] += adapted_param - original_params[name]

        # Apply Reptile update: θ ← θ + ε * (1/n) * Σ(θ̃ - θ)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(
                    original_params[name] +
                    self.outer_lr * param_updates[name] / len(tasks)
                )

        return total_query_loss / len(tasks)

    def adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        adaptation_steps: int = None
    ) -> nn.Module:
        """
        Adapt the meta-learned model to a new task.

        Args:
            support_data: Small amount of data from the new task
            adaptation_steps: Number of gradient steps (default: inner_steps)

        Returns:
            Adapted model
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        features, labels = support_data

        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            optimizer.step()

        return adapted_model


class TradingModel(nn.Module):
    """
    Simple neural network for trading signal prediction.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

### Data Preparation for Trading Tasks

```python
import numpy as np
import pandas as pd
from typing import Generator

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

    # Returns
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)

    # Moving averages
    features['sma_ratio'] = prices / prices.rolling(window).mean()
    features['ema_ratio'] = prices / prices.ewm(span=window).mean()

    # Volatility
    features['volatility'] = prices.pct_change().rolling(window).std()

    # Momentum
    features['momentum'] = prices / prices.shift(window) - 1

    # RSI
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features['rsi'] = 100 - (100 / (1 + gain / loss))

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

    # Random split point
    total_needed = support_size + query_size
    if len(aligned) < total_needed:
        raise ValueError(f"Not enough data: {len(aligned)} < {total_needed}")

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
    asset_data: dict,
    batch_size: int = 4
) -> Generator:
    """
    Generate batches of tasks from multiple assets.

    Args:
        asset_data: Dict of {asset_name: (prices, features)}
        batch_size: Number of tasks per batch

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

            # Create task data
            support, query = create_task_data(prices, features)
            tasks.append((support, query))

        yield tasks
```

---

## Implementation in Rust

The Rust implementation provides high-performance trading signal generation suitable for production environments.

### Project Structure

```
82_reptile_algorithm_trading/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── model/
│   │   ├── mod.rs
│   │   └── network.rs
│   ├── reptile/
│   │   ├── mod.rs
│   │   └── algorithm.rs
│   ├── data/
│   │   ├── mod.rs
│   │   ├── features.rs
│   │   └── bybit.rs
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── strategy.rs
│   │   └── signals.rs
│   └── backtest/
│       ├── mod.rs
│       └── engine.rs
├── examples/
│   ├── basic_reptile.rs
│   ├── multi_asset_training.rs
│   └── trading_strategy.rs
└── python/
    ├── reptile_trader.py
    ├── data_loader.py
    └── backtest.py
```

### Core Rust Implementation

See the `src/` directory for the complete Rust implementation with:

- High-performance matrix operations
- Async data fetching from Bybit
- Thread-safe model updates
- Production-ready error handling

---

## Practical Examples with Stock and Crypto Data

### Example 1: Multi-Asset Meta-Training

```python
import yfinance as yf

# Download data for multiple assets
assets = {
    'AAPL': yf.download('AAPL', period='2y'),
    'MSFT': yf.download('MSFT', period='2y'),
    'GOOGL': yf.download('GOOGL', period='2y'),
    'BTC-USD': yf.download('BTC-USD', period='2y'),
    'ETH-USD': yf.download('ETH-USD', period='2y'),
}

# Prepare data
asset_data = {}
for name, df in assets.items():
    prices = df['Close']
    features = create_trading_features(prices)
    asset_data[name] = (prices, features)

# Initialize model and Reptile trainer
model = TradingModel(input_size=8)  # 8 features
reptile = ReptileTrader(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    inner_steps=5
)

# Meta-training
task_gen = task_generator(asset_data, batch_size=4)
for epoch in range(1000):
    tasks = next(task_gen)
    loss = reptile.meta_train_step(tasks)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Query Loss: {loss:.6f}")
```

### Example 2: Rapid Adaptation to New Asset

```python
# New asset not seen during training
new_asset = yf.download('TSLA', period='1y')
new_prices = new_asset['Close']
new_features = create_trading_features(new_prices)

# Create small support set (just 20 samples)
support, query = create_task_data(new_prices, new_features, support_size=20)

# Adapt in just 5 gradient steps
adapted_model = reptile.adapt(support, adaptation_steps=5)

# Evaluate on query set
with torch.no_grad():
    predictions = adapted_model(query[0])
    loss = nn.MSELoss()(predictions, query[1])
    print(f"Adapted model query loss: {loss.item():.6f}")
```

### Example 3: Bybit Crypto Trading

```python
# Using Bybit data for cryptocurrency trading
import requests

def fetch_bybit_klines(symbol: str, interval: str = '1h', limit: int = 1000):
    """Fetch historical klines from Bybit."""
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.set_index('timestamp').sort_index()

    return df

# Fetch data for multiple crypto pairs
crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']
crypto_data = {}

for symbol in crypto_pairs:
    df = fetch_bybit_klines(symbol)
    prices = df['close']
    features = create_trading_features(prices)
    crypto_data[symbol] = (prices, features)

# Train on crypto data
crypto_task_gen = task_generator(crypto_data, batch_size=4)
for epoch in range(500):
    tasks = next(crypto_task_gen)
    loss = reptile.meta_train_step(tasks)
```

---

## Backtesting Framework

### Simple Backtest Implementation

```python
class ReptileBacktester:
    """
    Backtesting framework for Reptile-based trading strategies.
    """

    def __init__(
        self,
        reptile_trader: ReptileTrader,
        adaptation_window: int = 20,
        adaptation_steps: int = 5,
        prediction_threshold: float = 0.001
    ):
        self.reptile = reptile_trader
        self.adaptation_window = adaptation_window
        self.adaptation_steps = adaptation_steps
        self.threshold = prediction_threshold

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            prices: Price series
            features: Feature DataFrame
            initial_capital: Starting capital

        Returns:
            DataFrame with backtest results
        """
        results = []
        capital = initial_capital
        position = 0  # -1, 0, or 1

        feature_cols = list(features.columns)

        for i in range(self.adaptation_window, len(features) - 1):
            # Get adaptation data
            adapt_features = torch.FloatTensor(
                features.iloc[i-self.adaptation_window:i][feature_cols].values
            )
            adapt_returns = torch.FloatTensor(
                prices.pct_change().iloc[i-self.adaptation_window+1:i+1].values
            ).unsqueeze(1)

            # Adapt model
            adapted = self.reptile.adapt(
                (adapt_features[:-1], adapt_returns[:-1]),
                adaptation_steps=self.adaptation_steps
            )

            # Make prediction
            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = adapted(current_features).item()

            # Trading logic
            if prediction > self.threshold:
                new_position = 1  # Long
            elif prediction < -self.threshold:
                new_position = -1  # Short
            else:
                new_position = 0  # Neutral

            # Calculate returns
            actual_return = prices.iloc[i+1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'prediction': prediction,
                'actual_return': actual_return,
                'position': position,
                'position_return': position_return,
                'capital': capital
            })

            position = new_position

        return pd.DataFrame(results)
```

---

## Performance Evaluation

### Key Metrics

```python
def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate trading performance metrics.

    Args:
        results: Backtest results DataFrame

    Returns:
        Dictionary of metrics
    """
    returns = results['position_return']

    # Basic metrics
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Risk-adjusted metrics
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    sortino_ratio = np.sqrt(252) * returns.mean() / returns[returns < 0].std()

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(results[results['position'] != 0])
    }
```

### Expected Performance

| Metric | Target Range |
|--------|-------------|
| Sharpe Ratio | > 1.0 |
| Sortino Ratio | > 1.5 |
| Max Drawdown | < 20% |
| Win Rate | > 50% |

---

## Future Directions

### 1. Online Reptile

Continuously update the meta-initialization as new market data arrives:

```
θ ← θ + ε_t (θ̃_t - θ)
```

Where ε_t decreases over time to stabilize the initialization.

### 2. Hierarchical Tasks

Organize tasks hierarchically:
- Level 1: Different assets
- Level 2: Different market regimes
- Level 3: Different time scales

### 3. Uncertainty Quantification

Add Bayesian layers to quantify prediction uncertainty for risk management.

### 4. Multi-Objective Reptile

Optimize for multiple objectives simultaneously:
- Returns
- Risk (volatility)
- Maximum drawdown
- Transaction costs

---

## References

1. Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms. arXiv:1803.02999.
2. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
3. Antoniou, A., Edwards, H., & Storkey, A. (2019). How to train your MAML. ICLR.
4. Hospedales, T., et al. (2020). Meta-Learning in Neural Networks: A Survey. IEEE TPAMI.

---

## Running the Examples

### Python

```bash
# Navigate to chapter directory
cd 82_reptile_algorithm_trading

# Install dependencies
pip install torch numpy pandas yfinance scikit-learn

# Run Python examples
python python/reptile_trader.py
```

### Rust

```bash
# Navigate to chapter directory
cd 82_reptile_algorithm_trading

# Build the project
cargo build --release

# Run examples
cargo run --example basic_reptile
cargo run --example multi_asset_training
cargo run --example trading_strategy
```

---

## Summary

The Reptile algorithm offers a powerful yet simple approach to meta-learning for trading:

- **Simplicity**: First-order updates only, easy to implement
- **Efficiency**: Lower computational cost than MAML
- **Flexibility**: Works with any differentiable model
- **Fast Adaptation**: Few gradient steps to adapt to new conditions

By learning a good initialization across diverse trading tasks, Reptile enables rapid adaptation to new market conditions with minimal data - a crucial capability in the ever-changing financial markets.

---

*Previous Chapter: [Chapter 81: MAML for Trading](../81_maml_for_trading)*

*Next Chapter: [Chapter 83: Prototypical Networks for Finance](../83_prototypical_networks_finance)*
