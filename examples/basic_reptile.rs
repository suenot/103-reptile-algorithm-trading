//! Basic Reptile Meta-Learning Example
//!
//! This example demonstrates the core Reptile algorithm:
//! 1. Create a model and trainer
//! 2. Generate simulated tasks
//! 3. Run meta-training
//! 4. Adapt to a new task

use reptile_trading::prelude::*;
use reptile_trading::data::bybit::SimulatedDataSource;
use reptile_trading::reptile::algorithm::TaskData;

fn create_task_from_data(features: &[Vec<f64>], targets: &[f64]) -> TaskData {
    let split_idx = features.len() * 2 / 3;

    TaskData::new(
        features[..split_idx].to_vec(),
        targets[..split_idx].to_vec(),
        features[split_idx..].to_vec(),
        targets[split_idx..].to_vec(),
    )
}

fn main() {
    println!("=== Basic Reptile Meta-Learning Example ===\n");

    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create model
    println!("Creating trading model...");
    let model = TradingModel::new(8, 32, 1);
    println!("Model parameters: {}", model.num_parameters());

    // Create Reptile trainer
    println!("\nCreating Reptile trainer...");
    let mut trainer = ReptileTrainer::new(
        model,
        0.01,   // inner_lr
        0.001,  // outer_lr
        5,      // inner_steps
    );
    println!("Inner LR: {}", trainer.inner_lr());
    println!("Outer LR: {}", trainer.outer_lr());
    println!("Inner Steps: {}", trainer.inner_steps());

    // Generate simulated data for multiple "assets"
    println!("\nGenerating simulated market data...");
    let sources = vec![
        SimulatedDataSource::new("BTCUSDT", 50000.0, 0.02),
        SimulatedDataSource::new("ETHUSDT", 3000.0, 0.025),
        SimulatedDataSource::new("SOLUSDT", 100.0, 0.03),
    ];

    let feature_gen = FeatureGenerator::new(20);
    let mut all_tasks = Vec::new();

    for source in &sources {
        let klines = source.generate_klines(200);
        let features = feature_gen.generate_features(&klines);
        let targets = feature_gen.generate_targets(&klines, 5);

        if features.len() >= 50 && targets.len() >= 50 {
            let min_len = features.len().min(targets.len());
            let task = create_task_from_data(
                &features[..min_len],
                &targets[..min_len],
            );
            all_tasks.push(task);
        }
    }

    println!("Created {} tasks", all_tasks.len());

    // Meta-training loop
    println!("\n=== Meta-Training ===");
    let num_epochs = 50;

    for epoch in 0..num_epochs {
        let loss = trainer.meta_train_step(&all_tasks);

        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {}: Query Loss = {:.6}", epoch, loss);
        }
    }

    // Test adaptation to new task
    println!("\n=== Adaptation Test ===");
    let new_source = SimulatedDataSource::new("AVAXUSDT", 30.0, 0.035);
    let new_klines = new_source.generate_klines(100);
    let new_features = feature_gen.generate_features(&new_klines);
    let new_targets = feature_gen.generate_targets(&new_klines, 5);

    if new_features.len() >= 30 && new_targets.len() >= 30 {
        let support_features = &new_features[..20];
        let support_targets = &new_targets[..20];

        println!("Adapting to new asset with {} samples...", support_features.len());

        let adapted_model = trainer.adapt(
            support_features,
            support_targets,
            Some(10),  // 10 adaptation steps
        );

        // Make predictions
        println!("\nMaking predictions on new data:");
        let test_features = &new_features[20..25];
        let test_targets = &new_targets[20..25];

        for (i, (features, target)) in test_features.iter().zip(test_targets.iter()).enumerate() {
            let prediction = adapted_model.predict(features);
            let error = (prediction - target).abs();
            println!(
                "  Sample {}: Predicted={:.6}, Actual={:.6}, Error={:.6}",
                i + 1, prediction, target, error
            );
        }
    }

    println!("\n=== Example Complete ===");
}
