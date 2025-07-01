"""
Reptile Meta-Learning Algorithm for Trading

This module implements the Reptile meta-learning algorithm for algorithmic trading.
Reptile enables rapid adaptation to new market conditions with minimal data.

Reference: Nichol, A., Achiam, J., & Schulman, J. (2018).
"On First-Order Meta-Learning Algorithms." arXiv:1803.02999
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Generator
import copy
import numpy as np


class TradingModel(nn.Module):
    """
    Neural network for trading signal prediction.

    A simple feedforward network with ReLU activations
    suitable for meta-learning with the Reptile algorithm.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        """
        Initialize the trading model.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output predictions
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ReptileTrader:
    """
    Reptile meta-learning algorithm for trading strategy adaptation.

    Reptile learns a good initialization that can be quickly adapted
    to new tasks (assets, market regimes, etc.) with minimal data.

    Example:
        >>> model = TradingModel(input_size=8)
        >>> reptile = ReptileTrader(model, inner_lr=0.01, outer_lr=0.001, inner_steps=5)
        >>> # Meta-training
        >>> for epoch in range(1000):
        ...     tasks = generate_tasks(asset_data, batch_size=4)
        ...     loss = reptile.meta_train_step(tasks)
        >>> # Fast adaptation to new task
        >>> adapted_model = reptile.adapt(new_task_data, adaptation_steps=5)
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
        self.device = next(model.parameters()).device

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
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Perform k steps of SGD on the task
        adapted_model.train()
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            inner_optimizer.step()

        # Evaluate on query set
        adapted_model.eval()
        with torch.no_grad():
            query_features, query_labels = query_data
            query_features = query_features.to(self.device)
            query_labels = query_labels.to(self.device)
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

        The Reptile update rule:
            θ ← θ + ε * (1/n) * Σ(θ̃ - θ)

        where θ̃ represents parameters after k steps of SGD on a task.

        Args:
            tasks: List of (support_data, query_data) tuples

        Returns:
            Average query loss across tasks
        """
        if not tasks:
            return 0.0

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
        adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt the meta-learned model to a new task.

        Args:
            support_data: Small amount of data from the new task (features, labels)
            adaptation_steps: Number of gradient steps (default: inner_steps)

        Returns:
            Adapted model ready for prediction
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        features, labels = support_data
        features = features.to(self.device)
        labels = labels.to(self.device)

        adapted_model.train()
        for _ in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = nn.MSELoss()(predictions, labels)
            loss.backward()
            optimizer.step()

        adapted_model.eval()
        return adapted_model

    def save(self, path: str):
        """Save the meta-learned model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
        }, path)

    @classmethod
    def load(cls, path: str, model: nn.Module) -> 'ReptileTrader':
        """Load a saved Reptile trader."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(
            model=model,
            inner_lr=checkpoint['inner_lr'],
            outer_lr=checkpoint['outer_lr'],
            inner_steps=checkpoint['inner_steps'],
        )


def train_reptile(
    reptile: ReptileTrader,
    task_generator: Generator,
    num_epochs: int,
    log_interval: int = 100
) -> List[float]:
    """
    Meta-training loop for Reptile.

    Args:
        reptile: ReptileTrader instance
        task_generator: Generator yielding batches of tasks
        num_epochs: Number of meta-training epochs
        log_interval: How often to print progress

    Returns:
        List of average losses per epoch
    """
    losses = []

    for epoch in range(num_epochs):
        tasks = next(task_generator)
        loss = reptile.meta_train_step(tasks)
        losses.append(loss)

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, Query Loss: {loss:.6f}")

    return losses


if __name__ == "__main__":
    # Example usage
    print("Reptile Meta-Learning for Trading")
    print("=" * 50)

    # Create model
    model = TradingModel(input_size=8, hidden_size=64, output_size=1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer
    reptile = ReptileTrader(
        model=model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5
    )
    print(f"Inner LR: {reptile.inner_lr}")
    print(f"Outer LR: {reptile.outer_lr}")
    print(f"Inner Steps: {reptile.inner_steps}")

    # Create dummy task
    support_features = torch.randn(20, 8)
    support_labels = torch.randn(20, 1)
    query_features = torch.randn(10, 8)
    query_labels = torch.randn(10, 1)

    # Test meta-training step
    tasks = [
        ((support_features, support_labels), (query_features, query_labels)),
        ((support_features, support_labels), (query_features, query_labels)),
    ]

    print("\nMeta-training step...")
    loss = reptile.meta_train_step(tasks)
    print(f"Query loss: {loss:.6f}")

    # Test adaptation
    print("\nAdapting to new task...")
    adapted_model = reptile.adapt((support_features, support_labels), adaptation_steps=10)

    # Make prediction
    with torch.no_grad():
        prediction = adapted_model(query_features[:1])
        print(f"Prediction: {prediction.item():.6f}")

    print("\nDone!")
