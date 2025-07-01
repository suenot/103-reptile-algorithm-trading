//! Reptile meta-learning algorithm.
//!
//! Reptile is a first-order meta-learning algorithm that learns a good
//! initialization for fast adaptation to new tasks.
//!
//! Reference: Nichol, A., Achiam, J., & Schulman, J. (2018).
//! "On First-Order Meta-Learning Algorithms." arXiv:1803.02999

use crate::model::network::TradingModel;

/// Task data for meta-learning
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Support set features for adaptation
    pub support_features: Vec<Vec<f64>>,
    /// Support set labels
    pub support_labels: Vec<f64>,
    /// Query set features for evaluation
    pub query_features: Vec<Vec<f64>>,
    /// Query set labels
    pub query_labels: Vec<f64>,
}

impl TaskData {
    /// Create new task data
    pub fn new(
        support_features: Vec<Vec<f64>>,
        support_labels: Vec<f64>,
        query_features: Vec<Vec<f64>>,
        query_labels: Vec<f64>,
    ) -> Self {
        Self {
            support_features,
            support_labels,
            query_features,
            query_labels,
        }
    }
}

/// Reptile meta-learning trainer
#[derive(Debug)]
pub struct ReptileTrainer {
    /// The model being trained
    model: TradingModel,
    /// Inner loop learning rate (for task adaptation)
    inner_lr: f64,
    /// Outer loop learning rate (meta-learning rate, epsilon)
    outer_lr: f64,
    /// Number of SGD steps per task
    inner_steps: usize,
    /// Epsilon for numerical gradients
    gradient_epsilon: f64,
}

impl ReptileTrainer {
    /// Create a new Reptile trainer
    ///
    /// # Arguments
    /// * `model` - The trading model to train
    /// * `inner_lr` - Learning rate for task-specific adaptation
    /// * `outer_lr` - Meta-learning rate (epsilon in Reptile paper)
    /// * `inner_steps` - Number of SGD steps per task (k in Reptile paper)
    pub fn new(
        model: TradingModel,
        inner_lr: f64,
        outer_lr: f64,
        inner_steps: usize,
    ) -> Self {
        Self {
            model,
            inner_lr,
            outer_lr,
            inner_steps,
            gradient_epsilon: 1e-4,
        }
    }

    /// Set the gradient epsilon for numerical differentiation
    pub fn set_gradient_epsilon(&mut self, epsilon: f64) {
        self.gradient_epsilon = epsilon;
    }

    /// Perform inner loop adaptation on a single task
    ///
    /// Returns the adapted model and the query loss
    fn inner_loop(&self, task: &TaskData) -> (TradingModel, f64) {
        let mut adapted_model = self.model.clone_model();

        // Perform k steps of SGD on the support set
        for _ in 0..self.inner_steps {
            let gradients = adapted_model.compute_gradients(
                &task.support_features,
                &task.support_labels,
                self.gradient_epsilon,
            );
            adapted_model.sgd_step(&gradients, self.inner_lr);
        }

        // Evaluate on query set
        let query_predictions = adapted_model.predict_batch(&task.query_features);
        let query_loss = adapted_model.compute_loss(&query_predictions, &task.query_labels);

        (adapted_model, query_loss)
    }

    /// Perform one meta-training step using Reptile
    ///
    /// # Arguments
    /// * `tasks` - Batch of tasks for meta-training
    ///
    /// # Returns
    /// Average query loss across all tasks
    pub fn meta_train_step(&mut self, tasks: &[TaskData]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }

        // Store original parameters
        let original_params = self.model.get_parameters();

        // Accumulate parameter differences
        let num_params = original_params.len();
        let mut param_diff_sum = vec![0.0; num_params];
        let mut total_query_loss = 0.0;

        // Process each task
        for task in tasks {
            // Perform inner loop adaptation
            let (adapted_model, query_loss) = self.inner_loop(task);
            total_query_loss += query_loss;

            // Compute parameter difference (θ̃ - θ)
            let adapted_params = adapted_model.get_parameters();
            for (i, (adapted, original)) in adapted_params.iter()
                .zip(original_params.iter())
                .enumerate()
            {
                param_diff_sum[i] += adapted - original;
            }
        }

        // Apply Reptile update: θ ← θ + ε * (1/n) * Σ(θ̃ - θ)
        let num_tasks = tasks.len() as f64;
        let mut new_params = original_params.clone();
        for (i, diff) in param_diff_sum.iter().enumerate() {
            new_params[i] += self.outer_lr * diff / num_tasks;
        }

        self.model.set_parameters(&new_params);

        total_query_loss / num_tasks
    }

    /// Adapt the meta-learned model to a new task
    ///
    /// # Arguments
    /// * `support_features` - Features from the new task
    /// * `support_labels` - Labels from the new task
    /// * `adaptation_steps` - Number of gradient steps (default: inner_steps)
    ///
    /// # Returns
    /// Adapted model ready for prediction
    pub fn adapt(
        &self,
        support_features: &[Vec<f64>],
        support_labels: &[f64],
        adaptation_steps: Option<usize>,
    ) -> TradingModel {
        let steps = adaptation_steps.unwrap_or(self.inner_steps);
        let mut adapted_model = self.model.clone_model();

        for _ in 0..steps {
            let gradients = adapted_model.compute_gradients(
                support_features,
                support_labels,
                self.gradient_epsilon,
            );
            adapted_model.sgd_step(&gradients, self.inner_lr);
        }

        adapted_model
    }

    /// Get a reference to the current model
    pub fn model(&self) -> &TradingModel {
        &self.model
    }

    /// Get a mutable reference to the current model
    pub fn model_mut(&mut self) -> &mut TradingModel {
        &mut self.model
    }

    /// Get the inner learning rate
    pub fn inner_lr(&self) -> f64 {
        self.inner_lr
    }

    /// Get the outer learning rate
    pub fn outer_lr(&self) -> f64 {
        self.outer_lr
    }

    /// Get the number of inner steps
    pub fn inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Set the inner learning rate
    pub fn set_inner_lr(&mut self, lr: f64) {
        self.inner_lr = lr;
    }

    /// Set the outer learning rate
    pub fn set_outer_lr(&mut self, lr: f64) {
        self.outer_lr = lr;
    }

    /// Set the number of inner steps
    pub fn set_inner_steps(&mut self, steps: usize) {
        self.inner_steps = steps;
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Epoch number
    pub epoch: usize,
    /// Average query loss
    pub avg_loss: f64,
    /// Minimum loss in this epoch
    pub min_loss: f64,
    /// Maximum loss in this epoch
    pub max_loss: f64,
}

/// Meta-training loop with logging
pub fn train_reptile(
    trainer: &mut ReptileTrainer,
    task_generator: impl Iterator<Item = Vec<TaskData>>,
    num_epochs: usize,
    log_interval: usize,
) -> Vec<TrainingStats> {
    let mut stats_history = Vec::new();
    let mut task_iter = task_generator;

    for epoch in 0..num_epochs {
        if let Some(tasks) = task_iter.next() {
            let losses: Vec<f64> = tasks.iter().map(|t| {
                let (_, loss) = trainer.inner_loop(t);
                loss
            }).collect();

            let avg_loss = trainer.meta_train_step(&tasks);
            let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let stats = TrainingStats {
                epoch,
                avg_loss,
                min_loss,
                max_loss,
            };

            if epoch % log_interval == 0 {
                tracing::info!(
                    "Epoch {}: avg_loss={:.6}, min_loss={:.6}, max_loss={:.6}",
                    epoch, avg_loss, min_loss, max_loss
                );
            }

            stats_history.push(stats);
        } else {
            break;
        }
    }

    stats_history
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_task() -> TaskData {
        TaskData::new(
            vec![vec![0.1, 0.2, 0.3, 0.4]; 10],
            vec![0.5; 10],
            vec![vec![0.2, 0.3, 0.4, 0.5]; 5],
            vec![0.6; 5],
        )
    }

    #[test]
    fn test_reptile_trainer_creation() {
        let model = TradingModel::new(4, 16, 1);
        let trainer = ReptileTrainer::new(model, 0.01, 0.001, 5);

        assert_eq!(trainer.inner_lr(), 0.01);
        assert_eq!(trainer.outer_lr(), 0.001);
        assert_eq!(trainer.inner_steps(), 5);
    }

    #[test]
    fn test_inner_loop() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = ReptileTrainer::new(model, 0.01, 0.001, 3);
        let task = create_dummy_task();

        let (_adapted_model, loss) = trainer.inner_loop(&task);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_meta_train_step() {
        let model = TradingModel::new(4, 8, 1);
        let mut trainer = ReptileTrainer::new(model, 0.01, 0.001, 3);
        let tasks = vec![create_dummy_task(), create_dummy_task()];

        let loss = trainer.meta_train_step(&tasks);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_adapt() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = ReptileTrainer::new(model, 0.01, 0.001, 3);

        let features = vec![vec![0.1, 0.2, 0.3, 0.4]; 10];
        let labels = vec![0.5; 10];

        let adapted = trainer.adapt(&features, &labels, Some(5));
        let prediction = adapted.predict(&[0.1, 0.2, 0.3, 0.4]);
        assert!(prediction.is_finite());
    }
}
