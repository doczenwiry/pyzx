# PyZX - Python library for quantum circuit rewriting
#        and optimization using the ZX-calculus
# Copyright (C) 2026 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the implementation of the Phase-copy rule (hereinafter PCY) for the ZXW-Calculus
cfr. Section 2.2.2 in https://arxiv.org/pdf/2302.12135

"""

__all__ = [
    'check_phase_copy_forward',
]

from pyzx.utils import EdgeType, VertexType
from pyzx.graph.base import BaseGraph, VT, ET

# TODO: is it necessary to check that a W_OUTPUT is connected to exactly one W_INPUT ?
# TODO: is it necessary to check that a W_INPUT is connected to exactly two neighbors ?
def check_phase_copy_forward(g: BaseGraph[VT,ET], w: VT, z: VT) -> bool:
    """Checks if the PCY rule can be applied in a forward way to a pair of W and Z vertices."""

    # Both vertices must be from the graph
    if w not in g.vertices() or z not in g.vertices():
        return False

    # The vertices involved must be; one W_OUTPUT and one Z_BOX
    if g.type(w) != VertexType.W_OUTPUT or g.type(z) != VertexType.Z_BOX:
        return False

    # The vertices must be connected through a common W_INPUT by SIMPLE edges
    wi_z = set(filter(
        lambda nb : g.type(nb) == VertexType.W_INPUT and g.edge_type(g.edge(z,nb)) == EdgeType.SIMPLE,
        g.neighbors(z)
    ))
    wi_w = set(filter(
        lambda nb : g.type(nb) == VertexType.W_INPUT and g.edge_type(g.edge(w, nb)) == EdgeType.SIMPLE,
        g.neighbors(w)
    ))
    if wi_z.isdisjoint(wi_w):
        return False

    # The Pcy rule is applicable to w and z
    return True