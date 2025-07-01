//! Neural network implementation for trading predictions.
//!
//! This module provides a simple feedforward neural network
//! suitable for meta-learning with the Reptile algorithm.

use rand_distr::{Distribution, Normal};

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl Activation {
    /// Apply the activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
        }
    }

    /// Compute the derivative of the activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Linear => 1.0,
        }
    }
}

/// A dense (fully connected) layer
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Activation,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| normal.sample(&mut rng))
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        Self {
            weights,
            biases,
            activation,
            input_size,
            output_size,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size, "Input size mismatch");

        let mut output = vec![0.0; self.output_size];

        for i in 0..self.output_size {
            let mut sum = self.biases[i];
            for j in 0..self.input_size {
                sum += self.weights[i][j] * input[j];
            }
            output[i] = self.activation.apply(sum);
        }

        output
    }

    /// Get the number of parameters in this layer
    pub fn num_parameters(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }

    /// Get all parameters as a flat vector
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_parameters());
        for row in &self.weights {
            params.extend(row.iter());
        }
        params.extend(self.biases.iter());
        params
    }

    /// Set parameters from a flat vector
    pub fn set_parameters(&mut self, params: &[f64]) {
        assert_eq!(params.len(), self.num_parameters(), "Parameter count mismatch");

        let mut idx = 0;
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights[i][j] = params[idx];
                idx += 1;
            }
        }
        for i in 0..self.output_size {
            self.biases[i] = params[idx];
            idx += 1;
        }
    }
}

/// A multi-layer neural network for trading predictions
#[derive(Debug, Clone)]
pub struct TradingModel {
    layers: Vec<DenseLayer>,
    input_size: usize,
    output_size: usize,
}

impl TradingModel {
    /// Create a new trading model with the specified architecture
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `hidden_size` - Number of neurons in hidden layers
    /// * `output_size` - Number of output predictions
    ///
    /// # Example
    /// ```
    /// use reptile_trading::TradingModel;
    /// let model = TradingModel::new(8, 64, 1);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let layers = vec![
            DenseLayer::new(input_size, hidden_size, Activation::ReLU),
            DenseLayer::new(hidden_size, hidden_size, Activation::ReLU),
            DenseLayer::new(hidden_size, output_size, Activation::Linear),
        ];

        Self {
            layers,
            input_size,
            output_size,
        }
    }

    /// Create a model with custom layer sizes
    pub fn with_layers(input_size: usize, layer_sizes: &[usize], output_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        for &size in layer_sizes {
            layers.push(DenseLayer::new(prev_size, size, Activation::ReLU));
            prev_size = size;
        }

        layers.push(DenseLayer::new(prev_size, output_size, Activation::Linear));

        Self {
            layers,
            input_size,
            output_size,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Predict trading signal for a single sample
    pub fn predict(&self, features: &[f64]) -> f64 {
        let output = self.forward(features);
        output[0]
    }

    /// Batch prediction
    pub fn predict_batch(&self, features_batch: &[Vec<f64>]) -> Vec<f64> {
        features_batch.iter().map(|f| self.predict(f)).collect()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }

    /// Get all parameters as a flat vector
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_parameters());
        for layer in &self.layers {
            params.extend(layer.get_parameters());
        }
        params
    }

    /// Set parameters from a flat vector
    pub fn set_parameters(&mut self, params: &[f64]) {
        let mut idx = 0;
        for layer in &mut self.layers {
            let layer_params = layer.num_parameters();
            layer.set_parameters(&params[idx..idx + layer_params]);
            idx += layer_params;
        }
    }

    /// Clone the model
    pub fn clone_model(&self) -> Self {
        self.clone()
    }

    /// Compute MSE loss
    pub fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(predictions.len(), targets.len(), "Prediction/target size mismatch");

        let n = predictions.len() as f64;
        predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / n
    }

    /// Compute numerical gradients for a batch
    pub fn compute_gradients(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        epsilon: f64,
    ) -> Vec<f64> {
        let params = self.get_parameters();
        let mut gradients = vec![0.0; params.len()];

        for i in 0..params.len() {
            // Forward pass with param + epsilon
            let mut params_plus = params.clone();
            params_plus[i] += epsilon;
            let mut model_plus = self.clone();
            model_plus.set_parameters(&params_plus);
            let preds_plus = model_plus.predict_batch(features);
            let loss_plus = model_plus.compute_loss(&preds_plus, targets);

            // Forward pass with param - epsilon
            let mut params_minus = params.clone();
            params_minus[i] -= epsilon;
            let mut model_minus = self.clone();
            model_minus.set_parameters(&params_minus);
            let preds_minus = model_minus.predict_batch(features);
            let loss_minus = model_minus.compute_loss(&preds_minus, targets);

            // Numerical gradient
            gradients[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }

        gradients
    }

    /// Perform a single SGD update step
    pub fn sgd_step(&mut self, gradients: &[f64], learning_rate: f64) {
        let mut params = self.get_parameters();
        for (p, g) in params.iter_mut().zip(gradients.iter()) {
            *p -= learning_rate * g;
        }
        self.set_parameters(&params);
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get output size
    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = TradingModel::new(8, 64, 1);
        assert_eq!(model.input_size(), 8);
        assert_eq!(model.output_size(), 1);
    }

    #[test]
    fn test_forward_pass() {
        let model = TradingModel::new(4, 8, 1);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_parameters() {
        let model = TradingModel::new(4, 8, 1);
        let params = model.get_parameters();
        assert!(params.len() > 0);

        let mut model2 = TradingModel::new(4, 8, 1);
        model2.set_parameters(&params);
        let params2 = model2.get_parameters();

        for (p1, p2) in params.iter().zip(params2.iter()) {
            assert!((p1 - p2).abs() < 1e-10);
        }
    }
}
