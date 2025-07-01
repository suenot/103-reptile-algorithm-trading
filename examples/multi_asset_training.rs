//! Multi-Asset Meta-Training Example
//!
//! This example demonstrates training Reptile across multiple assets
//! to learn a model that can quickly adapt to any new asset.

use reptile_trading::prelude::*;
use reptile_trading::data::bybit::SimulatedDataSource;
use reptile_trading::reptile::algorithm::TaskData;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Asset configuration for simulated data
struct AssetConfig {
    symbol: &'static str,
    initial_price: f64,
    volatility: f64,
}

fn main() {
    println!("=== Multi-Asset Meta-Training Example ===\n");

    // Define multiple assets with different characteristics
    let assets = vec![
        // Large cap stocks (lower volatility)
        AssetConfig { symbol: "AAPL", initial_price: 180.0, volatility: 0.015 },
        AssetConfig { symbol: "MSFT", initial_price: 380.0, volatility: 0.016 },
        AssetConfig { symbol: "GOOGL", initial_price: 140.0, volatility: 0.018 },

        // Crypto (higher volatility)
        AssetConfig { symbol: "BTCUSDT", initial_price: 50000.0, volatility: 0.025 },
        AssetConfig { symbol: "ETHUSDT", initial_price: 3000.0, volatility: 0.030 },
        AssetConfig { symbol: "SOLUSDT", initial_price: 100.0, volatility: 0.040 },

        // Medium volatility
        AssetConfig { symbol: "TSLA", initial_price: 250.0, volatility: 0.035 },
        AssetConfig { symbol: "NVDA", initial_price: 500.0, volatility: 0.028 },
    ];

    // Generate data for each asset
    println!("Generating data for {} assets...", assets.len());
    let feature_gen = FeatureGenerator::new(20);
    let mut asset_data: Vec<(String, Vec<Vec<f64>>, Vec<f64>)> = Vec::new();

    for config in &assets {
        let source = SimulatedDataSource::new(config.symbol, config.initial_price, config.volatility);
        let klines = source.generate_klines(300);
        let features = feature_gen.generate_features(&klines);
        let targets = feature_gen.generate_targets(&klines, 5);

        if features.len() >= 50 {
            let min_len = features.len().min(targets.len());
            println!("  {} - {} samples (volatility: {:.1}%)",
                config.symbol, min_len, config.volatility * 100.0);
            asset_data.push((
                config.symbol.to_string(),
                features[..min_len].to_vec(),
                targets[..min_len].to_vec(),
            ));
        }
    }

    // Create model and trainer
    println!("\nCreating model and trainer...");
    let model = TradingModel::new(8, 64, 1);
    let mut trainer = ReptileTrainer::new(model, 0.01, 0.001, 5);

    // Meta-training loop
    println!("\n=== Meta-Training Phase ===");
    let num_epochs = 100;
    let batch_size = 4;
    let mut rng = thread_rng();

    for epoch in 0..num_epochs {
        // Sample random batch of tasks
        let mut tasks = Vec::new();
        let mut sampled_assets = asset_data.clone();
        sampled_assets.shuffle(&mut rng);

        for (_symbol, features, targets) in sampled_assets.iter().take(batch_size) {
            // Random split point for support/query
            let split = features.len() * 2 / 3;
            tasks.push(TaskData::new(
                features[..split].to_vec(),
                targets[..split].to_vec(),
                features[split..].to_vec(),
                targets[split..].to_vec(),
            ));
        }

        let loss = trainer.meta_train_step(&tasks);

        if epoch % 20 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:3}: Average Query Loss = {:.6}", epoch, loss);
        }
    }

    // Test adaptation on held-out asset
    println!("\n=== Adaptation Test ===");

    // Create a completely new asset not seen during training
    let test_assets = vec![
        AssetConfig { symbol: "ADAUSDT", initial_price: 0.5, volatility: 0.045 },
        AssetConfig { symbol: "DOGEUSDT", initial_price: 0.1, volatility: 0.060 },
    ];

    for test_config in &test_assets {
        println!("\nAdapting to new asset: {}", test_config.symbol);

        let source = SimulatedDataSource::new(
            test_config.symbol,
            test_config.initial_price,
            test_config.volatility,
        );
        let klines = source.generate_klines(100);
        let features = feature_gen.generate_features(&klines);
        let targets = feature_gen.generate_targets(&klines, 5);

        if features.len() >= 40 {
            // Use only 20 samples for adaptation (few-shot learning)
            let support_features = &features[..20];
            let support_targets = &targets[..20];

            // Test on remaining data
            let test_features = &features[20..40];
            let test_targets = &targets[20..40];

            // Adapt with different numbers of steps
            for &steps in &[1, 5, 10, 20] {
                let adapted = trainer.adapt(support_features, support_targets, Some(steps));

                // Calculate test loss
                let predictions: Vec<f64> = test_features.iter()
                    .map(|f| adapted.predict(f))
                    .collect();

                let mse: f64 = predictions.iter()
                    .zip(test_targets.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>() / predictions.len() as f64;

                println!("  {} adaptation steps: MSE = {:.6}", steps, mse);
            }
        }
    }

    // Compare adaptation speed vs training from scratch
    println!("\n=== Comparison: Meta-Learned vs Random Init ===");

    let test_source = SimulatedDataSource::new("XRPUSDT", 0.6, 0.035);
    let test_klines = test_source.generate_klines(100);
    let test_features = feature_gen.generate_features(&test_klines);
    let test_targets = feature_gen.generate_targets(&test_klines, 5);

    if test_features.len() >= 50 {
        let support = (&test_features[..30], &test_targets[..30]);
        let query = (&test_features[30..50], &test_targets[30..50]);

        // Meta-learned model adaptation
        let adapted_meta = trainer.adapt(
            support.0,
            support.1,
            Some(10),
        );

        let meta_preds: Vec<f64> = query.0.iter().map(|f| adapted_meta.predict(f)).collect();
        let meta_mse: f64 = meta_preds.iter()
            .zip(query.1.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / meta_preds.len() as f64;

        // Random initialization model
        let random_model = TradingModel::new(8, 64, 1);
        let random_trainer = ReptileTrainer::new(random_model, 0.01, 0.001, 10);

        // Train random model with same data
        let random_adapted = random_trainer.adapt(
            support.0,
            support.1,
            Some(10),
        );

        let random_preds: Vec<f64> = query.0.iter().map(|f| random_adapted.predict(f)).collect();
        let random_mse: f64 = random_preds.iter()
            .zip(query.1.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / random_preds.len() as f64;

        println!("\nResults after 10 gradient steps:");
        println!("  Meta-learned init: MSE = {:.6}", meta_mse);
        println!("  Random init:       MSE = {:.6}", random_mse);
        println!("  Improvement:       {:.1}%",
            (random_mse - meta_mse) / random_mse * 100.0);
    }

    println!("\n=== Example Complete ===");
}
