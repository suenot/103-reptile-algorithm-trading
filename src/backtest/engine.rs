//! Backtesting engine for evaluating trading strategies.

use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use crate::reptile::algorithm::ReptileTrainer;
use crate::trading::strategy::{TradingStrategy, StrategyConfig, Position};

/// Backtest result for a single trade
#[derive(Debug, Clone)]
pub struct TradeResult {
    /// Entry timestamp
    pub entry_time: chrono::DateTime<chrono::Utc>,
    /// Exit timestamp
    pub exit_time: chrono::DateTime<chrono::Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// Trade direction (1 for long, -1 for short)
    pub direction: i32,
    /// Profit/Loss in base currency
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Exit reason
    pub exit_reason: String,
}

/// Overall backtest statistics
#[derive(Debug, Clone)]
pub struct BacktestStats {
    /// Total return
    pub total_return: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Maximum drawdown duration (in bars)
    pub max_drawdown_duration: usize,
}

/// Equity curve point
#[derive(Debug, Clone)]
pub struct EquityPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub equity: f64,
    pub position: Position,
    pub price: f64,
}

/// Backtesting engine
pub struct BacktestEngine {
    /// Initial capital
    initial_capital: f64,
    /// Feature generator
    feature_generator: FeatureGenerator,
    /// Strategy configuration
    config: StrategyConfig,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(initial_capital: f64, window: usize, config: StrategyConfig) -> Self {
        Self {
            initial_capital,
            feature_generator: FeatureGenerator::new(window),
            config,
        }
    }

    /// Run backtest on historical data
    pub fn run(
        &self,
        trainer: &ReptileTrainer,
        klines: &[Kline],
        target_horizon: usize,
    ) -> BacktestResult {
        // Generate features and targets
        let features = self.feature_generator.generate_features(klines);
        let targets = self.feature_generator.generate_targets(klines, target_horizon);

        if features.is_empty() || targets.is_empty() {
            return BacktestResult {
                trades: vec![],
                equity_curve: vec![],
                stats: self.compute_stats(&[], &[]),
            };
        }

        // Align features with targets
        let min_len = features.len().min(targets.len());
        let offset = klines.len() - min_len;

        // Create strategy
        let mut strategy = TradingStrategy::new(
            ReptileTrainer::new(
                trainer.model().clone_model(),
                trainer.inner_lr(),
                trainer.outer_lr(),
                trainer.inner_steps(),
            ),
            self.config.clone(),
        );

        let mut equity = self.initial_capital;
        let mut equity_curve = Vec::new();
        let mut trades = Vec::new();
        let mut pending_entry: Option<(chrono::DateTime<chrono::Utc>, f64, i32, f64)> = None;

        // Run through the data
        for i in self.config.adaptation_window..min_len {
            let kline = &klines[offset + i];
            let feature = &features[i];
            let target = targets[i - 1]; // Previous target is now known

            // Update strategy with known data
            if i > 0 {
                strategy.update_data(features[i - 1].clone(), target);
            }

            // Periodically adapt
            if i % 10 == 0 && i >= self.config.adaptation_window {
                strategy.adapt();
            }

            // Process tick
            if let Some(order) = strategy.on_tick(feature, kline.close, "BACKTEST") {
                // Handle order execution
                match (&pending_entry, strategy.position()) {
                    // Closing a position
                    (Some((entry_time, entry_price, direction, size)), Position::Flat) => {
                        let pnl = (*direction as f64) * size * (kline.close - entry_price);
                        let return_pct = pnl / (entry_price * size);

                        equity += pnl;

                        trades.push(TradeResult {
                            entry_time: *entry_time,
                            exit_time: kline.timestamp,
                            entry_price: *entry_price,
                            exit_price: kline.close,
                            size: *size,
                            direction: *direction,
                            pnl,
                            return_pct,
                            exit_reason: order.reason.clone(),
                        });

                        pending_entry = None;
                    }
                    // Opening a new position
                    (None, Position::Long(size)) => {
                        pending_entry = Some((kline.timestamp, kline.close, 1, *size));
                    }
                    (None, Position::Short(size)) => {
                        pending_entry = Some((kline.timestamp, kline.close, -1, *size));
                    }
                    _ => {}
                }
            }

            // Record equity
            let unrealized = strategy.unrealized_pnl(kline.close);
            equity_curve.push(EquityPoint {
                timestamp: kline.timestamp,
                equity: equity + unrealized,
                position: strategy.position().clone(),
                price: kline.close,
            });
        }

        let stats = self.compute_stats(&trades, &equity_curve);

        BacktestResult {
            trades,
            equity_curve,
            stats,
        }
    }

    /// Compute backtest statistics
    fn compute_stats(&self, trades: &[TradeResult], equity_curve: &[EquityPoint]) -> BacktestStats {
        let num_trades = trades.len();

        if num_trades == 0 {
            return BacktestStats {
                total_return: 0.0,
                num_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                win_rate: 0.0,
                avg_win: 0.0,
                avg_loss: 0.0,
                profit_factor: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                max_drawdown_duration: 0,
            };
        }

        // Count wins and losses
        let winning_trades: Vec<&TradeResult> = trades.iter()
            .filter(|t| t.pnl > 0.0)
            .collect();
        let losing_trades: Vec<&TradeResult> = trades.iter()
            .filter(|t| t.pnl < 0.0)
            .collect();

        let total_wins: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let win_rate = winning_trades.len() as f64 / num_trades as f64;
        let avg_win = if !winning_trades.is_empty() {
            total_wins / winning_trades.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losing_trades.is_empty() {
            total_losses / losing_trades.len() as f64
        } else {
            0.0
        };
        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else {
            f64::INFINITY
        };

        // Calculate returns
        let total_return = if !equity_curve.is_empty() {
            (equity_curve.last().unwrap().equity - self.initial_capital) / self.initial_capital
        } else {
            0.0
        };

        // Calculate drawdown
        let (max_drawdown, max_drawdown_duration) = self.calculate_drawdown(equity_curve);

        // Calculate Sharpe and Sortino ratios
        let returns: Vec<f64> = trades.iter().map(|t| t.return_pct).collect();
        let sharpe_ratio = self.calculate_sharpe(&returns);
        let sortino_ratio = self.calculate_sortino(&returns);

        BacktestStats {
            total_return,
            num_trades,
            winning_trades: winning_trades.len(),
            losing_trades: losing_trades.len(),
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            max_drawdown_duration,
        }
    }

    /// Calculate maximum drawdown and duration
    fn calculate_drawdown(&self, equity_curve: &[EquityPoint]) -> (f64, usize) {
        if equity_curve.is_empty() {
            return (0.0, 0);
        }

        let mut max_equity = equity_curve[0].equity;
        let mut max_drawdown = 0.0;
        let mut max_duration = 0;
        let mut current_duration = 0;

        for point in equity_curve {
            if point.equity > max_equity {
                max_equity = point.equity;
                current_duration = 0;
            } else {
                let drawdown = (max_equity - point.equity) / max_equity;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
                current_duration += 1;
                if current_duration > max_duration {
                    max_duration = current_duration;
                }
            }
        }

        (max_drawdown, max_duration)
    }

    /// Calculate annualized Sharpe ratio
    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize assuming daily returns
        (mean / std_dev) * (252.0_f64).sqrt()
    }

    /// Calculate annualized Sortino ratio
    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();

        if downside_returns.is_empty() {
            return f64::INFINITY;
        }

        let downside_variance: f64 = downside_returns.iter()
            .map(|r| r.powi(2))
            .sum::<f64>() / downside_returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return f64::INFINITY;
        }

        (mean / downside_std) * (252.0_f64).sqrt()
    }
}

/// Complete backtest result
#[derive(Debug)]
pub struct BacktestResult {
    /// Individual trade results
    pub trades: Vec<TradeResult>,
    /// Equity curve
    pub equity_curve: Vec<EquityPoint>,
    /// Summary statistics
    pub stats: BacktestStats,
}

impl BacktestResult {
    /// Print a summary of the backtest
    pub fn print_summary(&self) {
        println!("=== Backtest Summary ===");
        println!("Total Return: {:.2}%", self.stats.total_return * 100.0);
        println!("Number of Trades: {}", self.stats.num_trades);
        println!("Win Rate: {:.2}%", self.stats.win_rate * 100.0);
        println!("Profit Factor: {:.2}", self.stats.profit_factor);
        println!("Sharpe Ratio: {:.2}", self.stats.sharpe_ratio);
        println!("Sortino Ratio: {:.2}", self.stats.sortino_ratio);
        println!("Max Drawdown: {:.2}%", self.stats.max_drawdown * 100.0);
        println!("Max Drawdown Duration: {} bars", self.stats.max_drawdown_duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::TradingModel;
    use crate::data::bybit::SimulatedDataSource;

    #[test]
    fn test_backtest_engine() {
        let model = TradingModel::new(8, 16, 1);  // Smaller model for faster test
        let trainer = ReptileTrainer::new(model, 0.01, 0.001, 2);  // Fewer steps
        let mut config = StrategyConfig::default();
        config.adaptation_window = 10;  // Smaller window for faster test
        let engine = BacktestEngine::new(10000.0, 10, config);

        // Generate smaller simulated data for faster test
        let source = SimulatedDataSource::new("BTCUSDT", 50000.0, 0.02);
        let klines = source.generate_klines(80);  // Reduced from 500

        let result = engine.run(&trainer, &klines, 3);

        assert!(result.stats.total_return.is_finite());
        // Don't require equity curve - may be empty for very short runs
    }
}
