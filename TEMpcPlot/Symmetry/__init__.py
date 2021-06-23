"""
Symmetry library





-----
"""
from .spacegroup import Spacegroup

from .CFML_exti import Search_Extinctions, Hkl_Ref_Conditions

from .GSASIIspc import spgbyNum

spgNum = {name: number for number, name in enumerate(spgbyNum[1:])}