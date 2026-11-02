"""The four job families. Importing this package registers each of them."""
from purse.families.aggregation import AGGREGATION
from purse.families.arithmetic import ARITHMETIC
from purse.families.grids import GRIDS
from purse.families.strings import STRINGS

__all__ = ["AGGREGATION", "ARITHMETIC", "GRIDS", "STRINGS"]
