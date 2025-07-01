"""
Backtesting framework for Reptile-based trading strategies.

This module provides utilities for evaluating trading strategies
using historical data and calculating performance metrics.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from reptile_trader import ReptileTrader, TradingModel
from data_loader import create_trading_features


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestStats:
    """Summary statistics for a backtest."""
    total_return: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int


class ReptileBacktester:
    """
    Backtesting framework for Reptile-based trading strategies.

    This backtester simulates trading using a Reptile meta-learned model,
    adapting to recent market data and generating trading signals.

    Example:
        >>> model = TradingModel(input_size=8)
        >>> reptile = ReptileTrader(model)
        >>> backtester = ReptileBacktester(reptile)
        >>> results = backtester.backtest(prices, features)
        >>> results.print_summary()
    """

    def __init__(
        self,
        reptile_trader: ReptileTrader,
        adaptation_window: int = 20,
        adaptation_steps: int = 5,
        prediction_threshold: float = 0.001,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03
    ):
        """
        Initialize the backtester.

        Args:
            reptile_trader: Trained ReptileTrader instance
            adaptation_window: Number of bars for adaptation data
            adaptation_steps: Number of gradient steps for adaptation
            prediction_threshold: Threshold for generating signals
            max_position_size: Maximum position size
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.reptile = reptile_trader
        self.adaptation_window = adaptation_window
        self.adaptation_steps = adaptation_steps
        self.threshold = prediction_threshold
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000.0,
        target_horizon: int = 5
    ) -> 'BacktestResult':
        """
        Run backtest on historical data.

        Args:
            prices: Price series
            features: Feature DataFrame
            initial_capital: Starting capital
            target_horizon: Prediction horizon for targets

        Returns:
            BacktestResult with trades and statistics
        """
        results = []
        trades = []
        capital = initial_capital

        position = 0  # -1, 0, or 1
        position_size = 0.0
        entry_price = None
        entry_time = None

        feature_cols = list(features.columns)

        # Create targets (known returns for adaptation)
        targets = prices.pct_change(target_horizon).shift(-target_horizon)

        for i in range(self.adaptation_window, len(features) - 1):
            current_time = features.index[i]
            current_price = prices.iloc[i]

            # Get adaptation data (known historical data)
            adapt_features = torch.FloatTensor(
                features.iloc[i-self.adaptation_window:i][feature_cols].values
            )
            adapt_returns = torch.FloatTensor(
                targets.iloc[i-self.adaptation_window:i].fillna(0).values
            ).unsqueeze(1)

            # Adapt model to recent data
            adapted = self.reptile.adapt(
                (adapt_features, adapt_returns),
                adaptation_steps=self.adaptation_steps
            )

            # Get current features and make prediction
            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = adapted(current_features).item()

            # Check stop loss / take profit
            if position != 0 and entry_price is not None:
                pnl_pct = position * (current_price - entry_price) / entry_price

                # Stop loss
                if pnl_pct < -self.stop_loss_pct:
                    trade_pnl = position * position_size * (current_price - entry_price)
                    trades.append(TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=position,
                        size=position_size,
                        pnl=trade_pnl,
                        return_pct=pnl_pct,
                        exit_reason='stop_loss'
                    ))
                    capital += trade_pnl
                    position = 0
                    position_size = 0.0
                    entry_price = None

                # Take profit
                elif pnl_pct > self.take_profit_pct:
                    trade_pnl = position * position_size * (current_price - entry_price)
                    trades.append(TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=position,
                        size=position_size,
                        pnl=trade_pnl,
                        return_pct=pnl_pct,
                        exit_reason='take_profit'
                    ))
                    capital += trade_pnl
                    position = 0
                    position_size = 0.0
                    entry_price = None

            # Trading logic based on prediction
            new_position = 0
            if prediction > self.threshold:
                new_position = 1  # Long
            elif prediction < -self.threshold:
                new_position = -1  # Short

            # Handle position changes
            if new_position != position:
                # Close existing position
                if position != 0 and entry_price is not None:
                    pnl_pct = position * (current_price - entry_price) / entry_price
                    trade_pnl = position * position_size * (current_price - entry_price)
                    trades.append(TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=position,
                        size=position_size,
                        pnl=trade_pnl,
                        return_pct=pnl_pct,
                        exit_reason='signal_change'
                    ))
                    capital += trade_pnl

                # Open new position
                if new_position != 0:
                    position = new_position
                    position_size = self.max_position_size
                    entry_price = current_price
                    entry_time = current_time
                else:
                    position = 0
                    position_size = 0.0
                    entry_price = None
                    entry_time = None

            # Record equity
            unrealized_pnl = 0.0
            if position != 0 and entry_price is not None:
                unrealized_pnl = position * position_size * (current_price - entry_price)

            results.append({
                'date': current_time,
                'price': current_price,
                'prediction': prediction,
                'position': position,
                'capital': capital,
                'equity': capital + unrealized_pnl,
            })

        results_df = pd.DataFrame(results)
        stats = self._calculate_stats(trades, results_df, initial_capital)

        return BacktestResult(
            trades=trades,
            equity_curve=results_df,
            stats=stats
        )

    def _calculate_stats(
        self,
        trades: List[TradeResult],
        results_df: pd.DataFrame,
        initial_capital: float
    ) -> BacktestStats:
        """Calculate backtest statistics."""

        if not trades:
            return BacktestStats(
                total_return=0.0,
                num_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0
            )

        # Win/loss statistics
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]

        total_wins = sum(t.pnl for t in winning)
        total_losses = sum(abs(t.pnl) for t in losing)

        win_rate = len(winning) / len(trades) if trades else 0
        avg_win = total_wins / len(winning) if winning else 0
        avg_loss = total_losses / len(losing) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Return calculation
        total_return = (results_df['equity'].iloc[-1] - initial_capital) / initial_capital

        # Drawdown calculation
        equity = results_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_drawdown = np.max(drawdowns)

        # Drawdown duration
        in_drawdown = drawdowns > 0
        drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]

        if len(drawdown_starts) > 0:
            if len(drawdown_ends) == 0 or drawdown_ends[-1] < drawdown_starts[-1]:
                drawdown_ends = np.append(drawdown_ends, len(drawdowns) - 1)
            max_drawdown_duration = max(drawdown_ends[i] - drawdown_starts[i]
                                        for i in range(min(len(drawdown_starts), len(drawdown_ends))))
        else:
            max_drawdown_duration = 0

        # Risk-adjusted returns
        returns = [t.return_pct for t in trades]
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            downside_returns = [r for r in returns if r < 0]

            sharpe_ratio = np.sqrt(252) * mean_return / std_return if std_return > 0 else 0
            sortino_ratio = (np.sqrt(252) * mean_return / np.std(downside_returns)
                           if downside_returns else float('inf'))
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        return BacktestStats(
            total_return=total_return,
            num_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration
        )


class BacktestResult:
    """Complete backtest result with trades and statistics."""

    def __init__(
        self,
        trades: List[TradeResult],
        equity_curve: pd.DataFrame,
        stats: BacktestStats
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.stats = stats

    def print_summary(self):
        """Print a summary of the backtest results."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Return:        {self.stats.total_return * 100:.2f}%")
        print(f"Number of Trades:    {self.stats.num_trades}")
        print(f"Winning Trades:      {self.stats.winning_trades}")
        print(f"Losing Trades:       {self.stats.losing_trades}")
        print(f"Win Rate:            {self.stats.win_rate * 100:.2f}%")
        print(f"Profit Factor:       {self.stats.profit_factor:.2f}")
        print(f"Sharpe Ratio:        {self.stats.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {self.stats.sortino_ratio:.2f}")
        print(f"Max Drawdown:        {self.stats.max_drawdown * 100:.2f}%")
        print(f"Max DD Duration:     {self.stats.max_drawdown_duration} bars")
        print("=" * 50)


if __name__ == "__main__":
    from data_loader import generate_simulated_data, create_trading_features

    print("Backtesting Reptile Trading Strategy")
    print("=" * 50)

    # Generate simulated data
    print("\nGenerating simulated data...")
    df = generate_simulated_data("BTCUSDT", initial_price=50000.0, num_periods=500)
    prices = df['close']
    features = create_trading_features(prices)

    print(f"Data points: {len(features)}")
    print(f"Features: {list(features.columns)}")

    # Create and train model
    print("\nCreating model...")
    model = TradingModel(input_size=8, hidden_size=32, output_size=1)
    reptile = ReptileTrader(
        model=model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5
    )

    # Create backtester
    print("\nRunning backtest...")
    backtester = ReptileBacktester(
        reptile_trader=reptile,
        adaptation_window=20,
        adaptation_steps=5,
        prediction_threshold=0.001
    )

    # Run backtest
    result = backtester.backtest(
        prices=prices.loc[features.index],
        features=features,
        initial_capital=10000.0
    )

    # Print results
    result.print_summary()

    # Show some trades
    if result.trades:
        print("\nSample Trades:")
        for trade in result.trades[:5]:
            print(f"  {trade.direction:+d} @ {trade.entry_price:.2f} -> {trade.exit_price:.2f} "
                  f"= {trade.pnl:.2f} ({trade.exit_reason})")

    print("\nDone!")
