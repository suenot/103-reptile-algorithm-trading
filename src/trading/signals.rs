//! Trading signal definitions and processing.

use std::fmt;

/// Trading signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// Neutral (no position)
    Neutral,
}

impl fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalDirection::Long => write!(f, "LONG"),
            SignalDirection::Short => write!(f, "SHORT"),
            SignalDirection::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// A trading signal with confidence and metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal direction
    pub direction: SignalDirection,
    /// Predicted return magnitude
    pub prediction: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp of the signal
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Associated symbol
    pub symbol: String,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        direction: SignalDirection,
        prediction: f64,
        confidence: f64,
        symbol: &str,
    ) -> Self {
        Self {
            direction,
            prediction,
            confidence,
            timestamp: chrono::Utc::now(),
            symbol: symbol.to_string(),
        }
    }

    /// Create a signal from a raw prediction value
    pub fn from_prediction(prediction: f64, threshold: f64, symbol: &str) -> Self {
        let direction = if prediction > threshold {
            SignalDirection::Long
        } else if prediction < -threshold {
            SignalDirection::Short
        } else {
            SignalDirection::Neutral
        };

        let confidence = (prediction.abs() / threshold).min(1.0);

        Self::new(direction, prediction, confidence, symbol)
    }

    /// Check if signal is actionable (not neutral)
    pub fn is_actionable(&self) -> bool {
        self.direction != SignalDirection::Neutral
    }

    /// Get the position size modifier (-1, 0, or 1)
    pub fn position_modifier(&self) -> i32 {
        match self.direction {
            SignalDirection::Long => 1,
            SignalDirection::Short => -1,
            SignalDirection::Neutral => 0,
        }
    }

    /// Get position size with confidence weighting
    pub fn weighted_position(&self) -> f64 {
        self.position_modifier() as f64 * self.confidence
    }
}

/// Signal aggregator for combining multiple signals
#[derive(Debug, Default)]
pub struct SignalAggregator {
    signals: Vec<TradingSignal>,
}

impl SignalAggregator {
    /// Create a new signal aggregator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a signal
    pub fn add_signal(&mut self, signal: TradingSignal) {
        self.signals.push(signal);
    }

    /// Get the consensus signal
    pub fn consensus(&self, symbol: &str) -> TradingSignal {
        if self.signals.is_empty() {
            return TradingSignal::new(SignalDirection::Neutral, 0.0, 0.0, symbol);
        }

        let total_weight: f64 = self.signals.iter()
            .map(|s| s.confidence)
            .sum();

        if total_weight == 0.0 {
            return TradingSignal::new(SignalDirection::Neutral, 0.0, 0.0, symbol);
        }

        let weighted_sum: f64 = self.signals.iter()
            .map(|s| s.weighted_position())
            .sum();

        let avg_prediction: f64 = self.signals.iter()
            .map(|s| s.prediction * s.confidence)
            .sum::<f64>() / total_weight;

        let consensus_value = weighted_sum / total_weight;

        let direction = if consensus_value > 0.3 {
            SignalDirection::Long
        } else if consensus_value < -0.3 {
            SignalDirection::Short
        } else {
            SignalDirection::Neutral
        };

        let confidence = consensus_value.abs();

        TradingSignal::new(direction, avg_prediction, confidence, symbol)
    }

    /// Clear all signals
    pub fn clear(&mut self) {
        self.signals.clear();
    }

    /// Get the number of signals
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        let signal = TradingSignal::from_prediction(0.05, 0.01, "BTCUSDT");
        assert_eq!(signal.direction, SignalDirection::Long);

        let signal = TradingSignal::from_prediction(-0.03, 0.01, "BTCUSDT");
        assert_eq!(signal.direction, SignalDirection::Short);

        let signal = TradingSignal::from_prediction(0.005, 0.01, "BTCUSDT");
        assert_eq!(signal.direction, SignalDirection::Neutral);
    }

    #[test]
    fn test_signal_aggregator() {
        let mut agg = SignalAggregator::new();

        agg.add_signal(TradingSignal::new(SignalDirection::Long, 0.02, 0.8, "BTC"));
        agg.add_signal(TradingSignal::new(SignalDirection::Long, 0.01, 0.6, "BTC"));
        agg.add_signal(TradingSignal::new(SignalDirection::Short, -0.01, 0.3, "BTC"));

        let consensus = agg.consensus("BTC");
        assert_eq!(consensus.direction, SignalDirection::Long);
    }
}
