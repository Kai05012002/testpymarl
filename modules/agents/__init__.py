REGISTRY = {}

from .rnn_agent import RNNAgent
from .gat_agent import GATAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["gat_agent"] = GATAgent
