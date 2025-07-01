//! Trading Strategy Example
//!
//! This example demonstrates using Reptile for adaptive trading,
//! including signal generation, position management, and backtesting.

use reptile_trading::prelude::*;
use reptile_trading::data::bybit::SimulatedDataSource;
use reptile_trading::reptile::algorithm::TaskData;
use reptile_trading::trading::strategy::{TradingStrategy, StrategyConfig};
use reptile_trading::backtest::engine::BacktestEngine;

fn main() {
    println!("=== Reptile Trading Strategy Example ===\n");

    // Step 1: Generate training data from multiple market regimes
    println!("Step 1: Generating multi-regime training data...");

    let feature_gen = FeatureGenerator::new(20);
    let mut all_tasks = Vec::new();

    // Simulate different market regimes
    let regimes = vec![
        ("Bull Market", 50000.0, 0.015, 0.0005),   // Low vol, upward drift
        ("Bear Market", 50000.0, 0.020, -0.0003),  // Medium vol, downward drift
        ("Sideways", 50000.0, 0.010, 0.0),         // Low vol, no drift
        ("High Volatility", 50000.0, 0.040, 0.0001), // High vol
    ];

    for (name, price, vol, _drift) in &regimes {
        let source = SimulatedDataSource::new(name, *price, *vol);
        let klines = source.generate_klines(200);
        let features = feature_gen.generate_features(&klines);
        let targets = feature_gen.generate_targets(&klines, 5);

        if features.len() >= 60 {
            let min_len = features.len().min(targets.len());
            let split = min_len * 2 / 3;

            all_tasks.push(TaskData::new(
                features[..split].to_vec(),
                targets[..split].to_vec(),
                features[split..min_len].to_vec(),
                targets[split..min_len].to_vec(),
            ));
            println!("  {} - {} samples", name, min_len);
        }
    }

    // Step 2: Meta-train the model
    println!("\nStep 2: Meta-training Reptile model...");

    let model = TradingModel::new(8, 64, 1);
    let mut trainer = ReptileTrainer::new(model, 0.01, 0.001, 5);

    for epoch in 0..50 {
        let loss = trainer.meta_train_step(&all_tasks);
        if epoch % 10 == 0 {
            println!("  Epoch {:2}: Loss = {:.6}", epoch, loss);
        }
    }

    // Step 3: Create trading strategy with the trained model
    println!("\nStep 3: Creating trading strategy...");

    let config = StrategyConfig {
        adaptation_window: 20,
        adaptation_steps: 5,
        prediction_threshold: 0.001,
        max_position_size: 1.0,
        stop_loss_pct: 0.02,
        take_profit_pct: 0.03,
    };

    println!("  Adaptation window: {} bars", config.adaptation_window);
    println!("  Prediction threshold: {:.4}", config.prediction_threshold);
    println!("  Stop loss: {:.1}%", config.stop_loss_pct * 100.0);
    println!("  Take profit: {:.1}%", config.take_profit_pct * 100.0);

    // Step 4: Backtest on new data
    println!("\nStep 4: Backtesting on unseen data...");

    let engine = BacktestEngine::new(10000.0, 20, config.clone());

    // Test on different assets
    let test_assets = vec![
        ("BTCUSDT", 50000.0, 0.025),
        ("ETHUSDT", 3000.0, 0.030),
        ("SOLUSDT", 100.0, 0.040),
    ];

    for (symbol, price, vol) in &test_assets {
        println!("\n--- {} ---", symbol);

        let source = SimulatedDataSource::new(symbol, *price, *vol);
        let klines = source.generate_klines(500);

        let result = engine.run(&trainer, &klines, 5);

        println!("  Total Return: {:.2}%", result.stats.total_return * 100.0);
        println!("  Trades: {} (Win rate: {:.1}%)",
            result.stats.num_trades,
            result.stats.win_rate * 100.0);
        println!("  Sharpe Ratio: {:.2}", result.stats.sharpe_ratio);
        println!("  Max Drawdown: {:.2}%", result.stats.max_drawdown * 100.0);
    }

    // Step 5: Demonstrate live adaptation
    println!("\n\nStep 5: Demonstrating live adaptation...");

    let mut strategy = TradingStrategy::new(
        ReptileTrainer::new(
            trainer.model().clone_model(),
            trainer.inner_lr(),
            trainer.outer_lr(),
            trainer.inner_steps(),
        ),
        config,
    );

    // Simulate live trading with streaming data
    let live_source = SimulatedDataSource::new("XRPUSDT", 0.6, 0.035);
    let live_klines = live_source.generate_klines(100);
    let live_features = feature_gen.generate_features(&live_klines);
    let live_targets = feature_gen.generate_targets(&live_klines, 5);

    println!("\nSimulating live trading for {} ticks...", live_features.len().min(50));

    let mut trades_executed = 0;
    let min_len = live_features.len().min(live_targets.len()).min(50);

    for i in 20..min_len {
        let feature = &live_features[i];
        let price = live_klines[live_klines.len() - min_len + i].close;

        // Update with known historical data
        if i > 0 {
            strategy.update_data(live_features[i - 1].clone(), live_targets[i - 1]);
        }

        // Adapt periodically
        if i % 10 == 0 {
            strategy.adapt();
        }

        // Process tick
        if let Some(order) = strategy.on_tick(feature, price, "XRPUSDT") {
            trades_executed += 1;
            if trades_executed <= 5 {
                println!("  Tick {:3}: {:?} @ ${:.4} ({})",
                    i,
                    order.side,
                    price,
                    order.reason);
            }
        }
    }

    println!("  ... {} total orders generated", trades_executed);

    // Step 6: Compare with non-adaptive baseline
    println!("\n\nStep 6: Comparison with non-adaptive baseline...");

    // Reptile-based strategy (already trained)
    let test_source = SimulatedDataSource::new("AVAXUSDT", 30.0, 0.035);
    let test_klines = test_source.generate_klines(500);

    // Adaptive strategy
    let adaptive_result = engine.run(&trainer, &test_klines, 5);

    // Non-adaptive (random init)
    let random_model = TradingModel::new(8, 64, 1);
    let random_trainer = ReptileTrainer::new(random_model, 0.01, 0.001, 5);
    let non_adaptive_result = engine.run(&random_trainer, &test_klines, 5);

    println!("\n  Metric                 Reptile    Random Init");
    println!("  ---------------------------------------------------");
    println!("  Total Return:          {:>7.2}%   {:>7.2}%",
        adaptive_result.stats.total_return * 100.0,
        non_adaptive_result.stats.total_return * 100.0);
    println!("  Win Rate:              {:>7.1}%   {:>7.1}%",
        adaptive_result.stats.win_rate * 100.0,
        non_adaptive_result.stats.win_rate * 100.0);
    println!("  Sharpe Ratio:          {:>7.2}    {:>7.2}",
        adaptive_result.stats.sharpe_ratio,
        non_adaptive_result.stats.sharpe_ratio);
    println!("  Max Drawdown:          {:>7.2}%   {:>7.2}%",
        adaptive_result.stats.max_drawdown * 100.0,
        non_adaptive_result.stats.max_drawdown * 100.0);

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Reptile learns a good initialization for fast adaptation");
    println!("  2. The strategy adapts to new market conditions automatically");
    println!("  3. Meta-trained models outperform random initialization");
    println!("  4. Continuous adaptation helps handle regime changes");
}
