//! Trading strategy implementation using Reptile-adapted models.

use crate::model::network::TradingModel;
use crate::reptile::algorithm::ReptileTrainer;
use crate::trading::signals::{TradingSignal, SignalDirection};

/// Position state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    Long(f64),   // Size
    Short(f64),  // Size
    Flat,
}

impl Position {
    pub fn size(&self) -> f64 {
        match self {
            Position::Long(s) | Position::Short(s) => *s,
            Position::Flat => 0.0,
        }
    }

    pub fn is_flat(&self) -> bool {
        matches!(self, Position::Flat)
    }
}

/// Trading strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Prediction threshold for generating signals
    pub prediction_threshold: f64,
    /// Maximum position size (in base currency units)
    pub max_position_size: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Number of adaptation steps when market conditions change
    pub adaptation_steps: usize,
    /// Window size for adaptation data
    pub adaptation_window: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            prediction_threshold: 0.001,
            max_position_size: 1.0,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.03,
            adaptation_steps: 5,
            adaptation_window: 20,
        }
    }
}

/// Reptile-based trading strategy
pub struct TradingStrategy {
    /// Reptile trainer with meta-learned model
    trainer: ReptileTrainer,
    /// Current adapted model for live trading
    adapted_model: TradingModel,
    /// Strategy configuration
    config: StrategyConfig,
    /// Current position
    position: Position,
    /// Entry price for current position
    entry_price: Option<f64>,
    /// Recent features for adaptation
    recent_features: Vec<Vec<f64>>,
    /// Recent targets for adaptation
    recent_targets: Vec<f64>,
}

impl TradingStrategy {
    /// Create a new trading strategy
    pub fn new(trainer: ReptileTrainer, config: StrategyConfig) -> Self {
        let adapted_model = trainer.model().clone_model();

        Self {
            trainer,
            adapted_model,
            config,
            position: Position::Flat,
            entry_price: None,
            recent_features: Vec::new(),
            recent_targets: Vec::new(),
        }
    }

    /// Update recent data for online adaptation
    pub fn update_data(&mut self, features: Vec<f64>, target: f64) {
        self.recent_features.push(features);
        self.recent_targets.push(target);

        // Keep only the most recent data
        while self.recent_features.len() > self.config.adaptation_window {
            self.recent_features.remove(0);
            self.recent_targets.remove(0);
        }
    }

    /// Adapt the model to recent data
    pub fn adapt(&mut self) {
        if self.recent_features.len() >= 10 {
            self.adapted_model = self.trainer.adapt(
                &self.recent_features,
                &self.recent_targets,
                Some(self.config.adaptation_steps),
            );
        }
    }

    /// Generate trading signal from current features
    pub fn generate_signal(&self, features: &[f64], symbol: &str) -> TradingSignal {
        let prediction = self.adapted_model.predict(features);
        TradingSignal::from_prediction(prediction, self.config.prediction_threshold, symbol)
    }

    /// Process a new market tick and potentially generate orders
    pub fn on_tick(
        &mut self,
        features: &[f64],
        current_price: f64,
        symbol: &str,
    ) -> Option<Order> {
        // Check stop loss / take profit
        if let Some(entry) = self.entry_price {
            let pnl_pct = match self.position {
                Position::Long(_) => (current_price - entry) / entry,
                Position::Short(_) => (entry - current_price) / entry,
                Position::Flat => 0.0,
            };

            // Stop loss
            if pnl_pct < -self.config.stop_loss_pct {
                return Some(self.close_position(current_price, "stop_loss"));
            }

            // Take profit
            if pnl_pct > self.config.take_profit_pct {
                return Some(self.close_position(current_price, "take_profit"));
            }
        }

        // Generate signal
        let signal = self.generate_signal(features, symbol);

        // Execute based on signal
        match (signal.direction, &self.position) {
            // Open long if signal is long and we're flat
            (SignalDirection::Long, Position::Flat) => {
                Some(self.open_long(current_price, signal.confidence))
            }
            // Open short if signal is short and we're flat
            (SignalDirection::Short, Position::Flat) => {
                Some(self.open_short(current_price, signal.confidence))
            }
            // Close long if signal turns short
            (SignalDirection::Short, Position::Long(_)) => {
                Some(self.close_position(current_price, "signal_reversal"))
            }
            // Close short if signal turns long
            (SignalDirection::Long, Position::Short(_)) => {
                Some(self.close_position(current_price, "signal_reversal"))
            }
            // Close if signal goes neutral
            (SignalDirection::Neutral, Position::Long(_) | Position::Short(_)) => {
                Some(self.close_position(current_price, "signal_neutral"))
            }
            // No action needed
            _ => None,
        }
    }

    /// Open a long position
    fn open_long(&mut self, price: f64, confidence: f64) -> Order {
        let size = self.config.max_position_size * confidence;
        self.position = Position::Long(size);
        self.entry_price = Some(price);

        Order {
            side: OrderSide::Buy,
            size,
            price,
            reason: "signal_long".to_string(),
        }
    }

    /// Open a short position
    fn open_short(&mut self, price: f64, confidence: f64) -> Order {
        let size = self.config.max_position_size * confidence;
        self.position = Position::Short(size);
        self.entry_price = Some(price);

        Order {
            side: OrderSide::Sell,
            size,
            price,
            reason: "signal_short".to_string(),
        }
    }

    /// Close the current position
    fn close_position(&mut self, price: f64, reason: &str) -> Order {
        let (side, size) = match self.position {
            Position::Long(s) => (OrderSide::Sell, s),
            Position::Short(s) => (OrderSide::Buy, s),
            Position::Flat => return Order {
                side: OrderSide::Buy,
                size: 0.0,
                price,
                reason: "already_flat".to_string(),
            },
        };

        self.position = Position::Flat;
        self.entry_price = None;

        Order {
            side,
            size,
            price,
            reason: reason.to_string(),
        }
    }

    /// Get current position
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Get entry price
    pub fn entry_price(&self) -> Option<f64> {
        self.entry_price
    }

    /// Get unrealized PnL
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        match (self.entry_price, &self.position) {
            (Some(entry), Position::Long(size)) => (current_price - entry) * size,
            (Some(entry), Position::Short(size)) => (entry - current_price) * size,
            _ => 0.0,
        }
    }
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order to be executed
#[derive(Debug, Clone)]
pub struct Order {
    pub side: OrderSide,
    pub size: f64,
    pub price: f64,
    pub reason: String,
}

impl Order {
    /// Calculate the value of the order
    pub fn value(&self) -> f64 {
        self.size * self.price
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::network::TradingModel;

    fn create_test_strategy() -> TradingStrategy {
        let model = TradingModel::new(4, 16, 1);
        let trainer = ReptileTrainer::new(model, 0.01, 0.001, 5);
        TradingStrategy::new(trainer, StrategyConfig::default())
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = create_test_strategy();
        assert!(strategy.position().is_flat());
    }

    #[test]
    fn test_signal_generation() {
        let strategy = create_test_strategy();
        let features = vec![0.1, 0.2, 0.3, 0.4];
        let signal = strategy.generate_signal(&features, "BTCUSDT");
        assert!(signal.prediction.is_finite());
    }

    #[test]
    fn test_update_data() {
        let mut strategy = create_test_strategy();

        for i in 0..30 {
            strategy.update_data(vec![i as f64 * 0.1; 4], 0.01);
        }

        assert_eq!(strategy.recent_features.len(), 20);
    }
}
