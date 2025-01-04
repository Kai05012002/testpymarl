from .basic_controller import BasicMAC
from .gat_mac import GATMAC

REGISTRY = dict()

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["gat_mac"] = GATMAC