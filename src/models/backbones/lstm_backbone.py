"""
LSTM-style backbone (CNN + linear encoder).

Re-exports the CWT-LSTM autoencoder backbone for registry compatibility.
The actual implementation lives in cwtlstm.CWT_LSTM_Autoencoder.
"""

from ..cwtlstm import CWT_LSTM_Autoencoder

__all__ = ["CWT_LSTM_Autoencoder"]
