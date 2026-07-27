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


import itertools
import unittest
import sys

from pyzx import EdgeType

if __name__ == '__main__':
    sys.path.append('..')
    sys.path.append('.')

from pyzx.graph import Graph
from pyzx.utils import VertexType, set_z_box_label
from pyzx.rewrite_rules.wrules_pcy import (
    check_phase_copy_forward
)


class TestCheckForwardPhaseCopy(unittest.TestCase):
    """Tests for check_phase_copy_forward."""

    @staticmethod
    def __prepare_phase_copy_forward_graph():
        g = Graph()

        i = g.add_vertex(qubit=0.5, row=0)
        z = g.add_vertex(ty=VertexType.Z_BOX, qubit=0.5, row=1)
        set_z_box_label(g, z, 1j)
        wi = g.add_vertex(ty=VertexType.W_INPUT, qubit=0.5, row=2)
        wo = g.add_vertex(ty=VertexType.W_OUTPUT, qubit=0.5, row=3)
        o0 = g.add_vertex(qubit=0, row=4)
        o1 = g.add_vertex(qubit=1, row=4)

        g.add_edge((i, z))
        g.add_edge((z, wi))
        g.add_edge((wi, wo))
        g.add_edge((wo, o0))
        g.add_edge((wo, o1))

        return g, wo, wi, z

    def test_pcy_pair_phase_free(self):
        """Z-W pair with zero phase should match."""
        g, wo, wi, z = self.__prepare_phase_copy_forward_graph()
        self.assertTrue(check_phase_copy_forward(g, wo, z))

    def test_pcy_pair_phase_nonzero(self):
        """Z-W pair with any phase should match."""
        g, wo, wi, z = self.__prepare_phase_copy_forward_graph()
        set_z_box_label(g, z, 1j)
        self.assertTrue(check_phase_copy_forward(g, wo, z))

    def test_pcy_pair_missing_edge_one(self):
        """Z-W pair not connected through a W_INPUT should not match."""
        g, wo, wi, z = self.__prepare_phase_copy_forward_graph()
        g.remove_edge(g.edge(wo, wi))
        self.assertFalse(check_phase_copy_forward(g, wo, z))

    def test_pcy_pair_missing_edge_two(self):
        """Z-W pair not connected through a W_INPUT should not match."""
        g, wo, wi, z = self.__prepare_phase_copy_forward_graph()
        g.remove_edge(g.edge(wi, z))
        self.assertFalse(check_phase_copy_forward(g, wo, z))

    def test_zw_pairs_hadamard_edge_one(self):
        """Z-W pair connected through a W_INPUT with HADAMARD edges should not match."""
        g, wo, wi, z = self.__prepare_phase_copy_forward_graph()
        g.set_edge_type(g.edge(wo, wi), EdgeType.HADAMARD)
        self.assertFalse(check_phase_copy_forward(g, wo, z))

    def test_zw_pairs_hadamard_edge_two(self):
        """Z-W pair connected through a W_INPUT with HADAMARD edges should not match."""
        g, wo, wi, z = self.__prepare_phase_copy_forward_graph()
        g.set_edge_type(g.edge(wi, z), EdgeType.HADAMARD)
        self.assertFalse(check_phase_copy_forward(g, wo, z))

if __name__ == '__main__':
    unittest.main()
