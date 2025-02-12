REGISTRY = {}

from .rnn_agent import RNNAgent
from .gat_agent import GATAgent
from .enhanced_gat_agent import EnhancedGATAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["gat_agent"] = GATAgent
REGISTRY["enhanced_gat_agent"] = EnhancedGATAgent
