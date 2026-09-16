"""
test_end_to_end.py -- Unit tests for the End-to-End PyTorch components.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

# Adjust this import based on your directory structure
from end_to_end import EndToEndNetwork, CalibratedFocalLoss

class TestEndToEndComponents(unittest.TestCase):

    def setUp(self):
        # 11 features: 8 from STAGE1_FEATURES + 3 from MARKET_FEATURES (example)
        self.input_dim = 11 
        self.batch_size = 16
        self.net = EndToEndNetwork(input_dim=self.input_dim, hidden_dim=64)
        
        # Create dummy data
        self.X_dummy = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        self.y_dummy = torch.randint(0, 3, (self.batch_size,))

    def test_network_output_shape(self):
        """Ensure the network outputs logits for 3 classes (H, D, A)."""
        logits = self.net(self.X_dummy)
        self.assertEqual(logits.shape, (self.batch_size, 3))

    def test_gradient_flow(self):
        """Ensure gradients propagate through the network back to inputs."""
        criterion = CalibratedFocalLoss(gamma=2.0)
        logits = self.net(self.X_dummy)
        
        loss = criterion(logits, self.y_dummy)
        loss.backward()
        
        # Check that gradients exist for the first layer's weights
        self.assertIsNotNone(self.net.net[0].weight.grad)
        # Check that gradients flowed all the way back to the input tensor
        self.assertIsNotNone(self.X_dummy.grad)
        self.assertFalse(torch.isnan(loss).item())

    def test_focal_loss_shrinkage(self):
        """
        Mathematically verify that Focal Loss shrinks the gradient magnitude 
        for highly confident predictions compared to standard CrossEntropy.
        """
        ce_criterion = nn.CrossEntropyLoss(reduction='none')
        focal_criterion = CalibratedFocalLoss(gamma=2.0)
        
        # Create a highly confident prediction (Logits: [10.0, 0.0, 0.0])
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        target = torch.tensor([0])
        
        ce_loss = ce_criterion(logits, target).item()
        focal_loss = focal_criterion(logits, target).item()
        
        # Focal loss should be significantly smaller than CE loss for confident predictions
        self.assertLess(focal_loss, ce_loss)
        self.assertAlmostEqual(focal_loss, 0.0, places=5)

    def test_probability_normalization(self):
        """Ensure the softmax conversion outputs valid probabilities summing to 1."""
        logits = self.net(self.X_dummy)
        probs = torch.softmax(logits, dim=1).detach().numpy()
        
        # Sum across the 3 outcome classes should equal 1.0 for every batch item
        sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()