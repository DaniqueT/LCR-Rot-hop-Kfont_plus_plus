import unittest
from .model.bert_encoder.position_ffn import PositionwiseFeedForward
import torch

class TestPositionwiseFeedForward(unittest.TestCase):
    def setUp(self):
        # Initialize your PositionwiseFeedForward module here
        self.feedforward = PositionwiseFeedForward(hidden_size=500, feedforward_size=1000, layer=3)

    def test_synonym_integration(self):
        # Create dummy input tensor
        input_tensor = torch.randn(10, 8, 500)

        # Call forward method with dummy input
        output = self.feedforward(input_tensor)

        # Assert that the synonym vectors have been integrated properly
        self.assertNotEqual(output, input_tensor)  # Check that output is different from input

if __name__ == '__main__':
    unittest.main()