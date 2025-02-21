import pytest
from models.train_deep_model import train_lstm_model
import os

def test_lstm_training():
    """Test that the LSTM model trains without errors."""
    train_lstm_model("AAPL")
    assert os.path.exists("models/saved_models/momentum_model_lstm"), "LSTM model was not saved!"
