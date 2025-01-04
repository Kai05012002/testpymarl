REGISTRY = {}

from .rnn_agent import RNNAgent
from .gat_agent import SimpleGAT

REGISTRY["rnn"] = RNNAgent
REGISTRY["simple_gat"] = SimpleGAT
