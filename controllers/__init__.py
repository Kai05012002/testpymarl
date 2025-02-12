from .basic_controller import BasicMAC
from .gat_mac import GATMAC
from .enhanced_gat_mac import EnhancedGATMAC

REGISTRY = dict()

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["gat_mac"] = GATMAC
REGISTRY["enhanced_gat_mac"] = EnhancedGATMAC
