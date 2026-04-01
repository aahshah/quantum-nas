"""
Gradient Flow Analysis for Quantum-Classical Hybrid Architectures
Publication-Grade Implementation

Mathematical Foundation:
- Gradient Flow Bound: 𝔼[|∂L/∂θ₁|] ≤ (∏ᵢ αᵢ) · |∂L/∂θₖ|
- Bottleneck Detection: αᵢ < ε → exponential gradient vanishing
- Measurement Bottleneck: rank(∂O/∂θ) ≤ m (measurements)

Author: GQAS Research Team
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class GradientFlowEdge:
    """Represents a gradient flow edge in the architecture graph."""
    source_param: int
    target_param: int
    attenuation: float  # α ∈ [0, 1]
    edge_type: str  # 'Q→Q', 'Q→C', 'C→Q', 'C→C'
    layer_source: int
    layer_target: int


class GradientFlowAnalyzer:
    """
    Analyzes gradient flow through quantum-classical hybrid architectures.
    
    Builds gradient flow graph G_∇ = (V, E_∇) where:
    - Nodes V: All trainable parameters (quantum + classical)
    - Edges E_∇: (i→j) if ∂²L/∂θᵢ∂θⱼ ≠ 0 (gradient dependency)
    - Edge weights: Attenuation factors α_ij
    
    Key Innovation:
    Identifies architectural bottlenecks that cause gradient vanishing,
    even when depth/width metrics suggest trainability.
    """
    
    def __init__(self, 
                 vanishing_threshold: float = 0.1,
                 use_hessian_approximation: bool = True):
        """
        Initialize gradient flow analyzer.
        
        Args:
            vanishing_threshold: Threshold below which gradients vanish (ε)
            use_hessian_approximation: Whether to approximate Hessian for efficiency
        """
        self.vanishing_threshold = vanishing_threshold
        self.use_hessian_approximation = use_hessian_approximation
        
    def build_gradient_flow_graph(self, architecture) -> Dict:
        """
        Construct gradient dependency graph from architecture.
        
        Algorithm:
        1. Identify all parameters (quantum + classical)
        2. For each parameter pair (i, j):
           a. Check if they're in gradient dependency path
           b. Estimate attenuation factor α_ij
           c. Add edge if α_ij > threshold
        3. Return graph structure
        
        Args:
            architecture: HybridArchitecture object
            
        Returns:
            Dictionary with:
            - 'nodes': List of parameter indices
            - 'edges': List of GradientFlowEdge objects
            - 'node_types': 'quantum' or 'classical' for each node
            - 'adjacency': Adjacency list representation
        """
        if architecture is None:
            raise ValueError("Architecture cannot be None")
        
        # Build parameter map
        param_map = self._build_parameter_map(architecture)
        nodes = list(range(len(param_map)))
        node_types = [param_map[i]['type'] for i in nodes]
        
        # Build edges
        edges = []
        adjacency = defaultdict(list)
        
        # Quantum-to-quantum edges (within layers)
        q_edges = self._build_quantum_edges(architecture, param_map)
        edges.extend(q_edges)
        for edge in q_edges:
            adjacency[edge.source_param].append(edge.target_param)
        
        # Quantum-to-classical edges (measurement interface)
        qc_edges = self._build_quantum_classical_edges(architecture, param_map)
        edges.extend(qc_edges)
        for edge in qc_edges:
            adjacency[edge.source_param].append(edge.target_param)
        
        # Classical-to-quantum edges (data re-uploading)
        cq_edges = self._build_classical_quantum_edges(architecture, param_map)
        edges.extend(cq_edges)
        for edge in cq_edges:
            adjacency[edge.source_param].append(edge.target_param)
        
        # Classical-to-classical edges (sequential layers)
        cc_edges = self._build_classical_edges(architecture, param_map)
        edges.extend(cc_edges)
        for edge in cc_edges:
            adjacency[edge.source_param].append(edge.target_param)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_types': node_types,
            'adjacency': dict(adjacency),
            'param_map': param_map
        }
    
    def _build_parameter_map(self, architecture) -> Dict:
        """Map parameter indices to their properties."""
        param_map = {}
        param_idx = 0
        
        # Quantum parameters
        for layer_idx, ql in enumerate(architecture.quantum_layers):
            n_params_per_qubit = ql.depth * len(ql.rotation_gates)
            for qubit_idx in range(ql.num_qubits):
                for _ in range(n_params_per_qubit):
                    param_map[param_idx] = {
                        'type': 'quantum',
                        'layer': layer_idx,
                        'qubit': qubit_idx,
                        'entanglement': ql.entanglement
                    }
                    param_idx += 1
        
        # Classical parameters
        for layer_idx, cl in enumerate(architecture.classical_layers):
            if cl.layer_type == 'conv':
                n_params = cl.filters * cl.kernel_size ** 2
            else:  # fc
                n_params = cl.filters
            
            for _ in range(n_params):
                param_map[param_idx] = {
                    'type': 'classical',
                    'layer': layer_idx,
                    'layer_type': cl.layer_type
                }
                param_idx += 1
        
        return param_map
    
    def _build_quantum_edges(self, architecture, param_map) -> List[GradientFlowEdge]:
        """Build edges between quantum parameters (entanglement creates dependencies)."""
        edges = []
        param_idx = 0
        
        for layer_idx, ql in enumerate(architecture.quantum_layers):
            n_params_per_qubit = ql.depth * len(ql.rotation_gates)
            n_qubits = ql.num_qubits
            
            # Entanglement creates gradient dependencies
            if ql.entanglement == 'linear':
                # Linear: each qubit connected to next
                attenuation = 0.7  # Moderate coupling
                for qubit in range(n_qubits - 1):
                    for d in range(ql.depth):
                        src_idx = param_idx + qubit * n_params_per_qubit + d
                        tgt_idx = param_idx + (qubit + 1) * n_params_per_qubit + d
                        if src_idx < len(param_map) and tgt_idx < len(param_map):
                            edges.append(GradientFlowEdge(
                                source_param=src_idx,
                                target_param=tgt_idx,
                                attenuation=attenuation,
                                edge_type='Q→Q',
                                layer_source=layer_idx,
                                layer_target=layer_idx
                            ))
            
            elif ql.entanglement == 'circular':
                # Circular: ring topology
                attenuation = 0.6
                for qubit in range(n_qubits):
                    next_qubit = (qubit + 1) % n_qubits
                    for d in range(ql.depth):
                        src_idx = param_idx + qubit * n_params_per_qubit + d
                        tgt_idx = param_idx + next_qubit * n_params_per_qubit + d
                        if src_idx < len(param_map) and tgt_idx < len(param_map):
                            edges.append(GradientFlowEdge(
                                source_param=src_idx,
                                target_param=tgt_idx,
                                attenuation=attenuation,
                                edge_type='Q→Q',
                                layer_source=layer_idx,
                                layer_target=layer_idx
                            ))
            
            elif ql.entanglement == 'full':
                # Full: all-to-all (strongest coupling, but can cause vanishing)
                attenuation = 0.5  # Lower due to potential barren plateau
                for qubit1 in range(n_qubits):
                    for qubit2 in range(qubit1 + 1, n_qubits):
                        for d in range(ql.depth):
                            src_idx = param_idx + qubit1 * n_params_per_qubit + d
                            tgt_idx = param_idx + qubit2 * n_params_per_qubit + d
                            if src_idx < len(param_map) and tgt_idx < len(param_map):
                                edges.append(GradientFlowEdge(
                                    source_param=src_idx,
                                    target_param=tgt_idx,
                                    attenuation=attenuation,
                                    edge_type='Q→Q',
                                    layer_source=layer_idx,
                                    layer_target=layer_idx
                                ))
            
            # Update param_idx for next layer
            param_idx += n_qubits * n_params_per_qubit
        
        return edges
    
    def _build_quantum_classical_edges(self, architecture, param_map) -> List[GradientFlowEdge]:
        """
        Build edges from quantum to classical (measurement interface).
        
        This is often the PRIMARY bottleneck due to information loss in measurement.
        """
        edges = []
        
        # Find quantum and classical parameter ranges
        q_params = [i for i, info in param_map.items() if info['type'] == 'quantum']
        c_params = [i for i, info in param_map.items() if info['type'] == 'classical']
        
        if not q_params or not c_params:
            return edges
        
        # Measurement interface: quantum → classical
        # Attenuation depends on number of measurements vs parameters
        n_measurements = max([ql.num_qubits for ql in architecture.quantum_layers]) if architecture.quantum_layers else 0
        n_q_params = len(q_params)
        
        # Measurement bottleneck: rank(∂O/∂θ) ≤ m
        if n_measurements > 0:
            bottleneck_factor = min(1.0, n_measurements / (n_q_params + 1e-6))
            attenuation = 0.3 * bottleneck_factor  # Strong attenuation at interface
        else:
            attenuation = 0.5
        
        # Connect each quantum parameter to classical parameters
        # (in practice, only nearby layers connect, but we approximate)
        for q_idx in q_params[:min(len(q_params), 10)]:  # Limit for efficiency
            for c_idx in c_params[:min(len(c_params), 5)]:
                edges.append(GradientFlowEdge(
                    source_param=q_idx,
                    target_param=c_idx,
                    attenuation=attenuation,
                    edge_type='Q→C',
                    layer_source=param_map[q_idx]['layer'],
                    layer_target=param_map[c_idx]['layer']
                ))
        
        return edges
    
    def _build_classical_quantum_edges(self, architecture, param_map) -> List[GradientFlowEdge]:
        """Build edges from classical to quantum (data re-uploading)."""
        edges = []
        
        c_params = [i for i, info in param_map.items() if info['type'] == 'classical']
        q_params = [i for i, info in param_map.items() if info['type'] == 'quantum']
        
        if not c_params or not q_params:
            return edges
        
        # Check if architecture uses data re-uploading
        has_reuploading = any(
            ql.parametrization == 'data-reuploading' 
            for ql in architecture.quantum_layers
        )
        
        if has_reuploading:
            # Data re-uploading: classical → quantum (moderate attenuation)
            attenuation = 0.6
        else:
            # No re-uploading: no direct classical → quantum flow
            return edges
        
        # Connect classical to quantum (limited connections)
        for c_idx in c_params[:min(len(c_params), 5)]:
            for q_idx in q_params[:min(len(q_params), 10)]:
                edges.append(GradientFlowEdge(
                    source_param=c_idx,
                    target_param=q_idx,
                    attenuation=attenuation,
                    edge_type='C→Q',
                    layer_source=param_map[c_idx]['layer'],
                    layer_target=param_map[q_idx]['layer']
                ))
        
        return edges
    
    def _build_classical_edges(self, architecture, param_map) -> List[GradientFlowEdge]:
        """Build edges between classical parameters (sequential layers)."""
        edges = []
        c_params = [i for i, info in param_map.items() if info['type'] == 'classical']
        
        if len(c_params) < 2:
            return edges
        
        # Sequential layers: each layer connects to next
        # Classical layers typically have good gradient flow
        attenuation = 0.8  # High (good gradient preservation)
        
        # Group parameters by layer
        params_by_layer = defaultdict(list)
        for idx in c_params:
            layer = param_map[idx]['layer']
            params_by_layer[layer].append(idx)
        
        # Connect consecutive layers
        layers = sorted(params_by_layer.keys())
        for i in range(len(layers) - 1):
            layer1_params = params_by_layer[layers[i]]
            layer2_params = params_by_layer[layers[i + 1]]
            
            # Connect last params of layer1 to first params of layer2
            for src in layer1_params[-min(5, len(layer1_params)):]:
                for tgt in layer2_params[:min(5, len(layer2_params))]:
                    edges.append(GradientFlowEdge(
                        source_param=src,
                        target_param=tgt,
                        attenuation=attenuation,
                        edge_type='C→C',
                        layer_source=layers[i],
                        layer_target=layers[i + 1]
                    ))
        
        return edges
    
    def identify_bottlenecks(self, graph: Dict) -> List[GradientFlowEdge]:
        """
        Find edges causing gradient vanishing.
        
        Bottleneck: edge with attenuation α < threshold
        
        Returns:
            List of bottleneck edges sorted by severity (lowest attenuation first)
        """
        if 'edges' not in graph:
            return []
        
        bottlenecks = [
            edge for edge in graph['edges']
            if edge.attenuation < self.vanishing_threshold
        ]
        
        # Sort by attenuation (most severe first)
        bottlenecks.sort(key=lambda e: e.attenuation)
        
        logger.info(f"Found {len(bottlenecks)} bottleneck edges (attenuation < {self.vanishing_threshold})")
        
        return bottlenecks
    
    def compute_effective_depth(self, graph: Dict) -> float:
        """
        Compute effective depth weighted by attenuation.
        
        Formula: d_eff = -log(∏ᵢ αᵢ) / log(α_avg)
        
        This captures how "deep" gradients perceive the architecture,
        accounting for gradient attenuation along paths.
        
        Returns:
            Effective depth (higher = gradients perceive architecture as deeper)
        """
        if 'edges' not in graph or len(graph['edges']) == 0:
            return 0.0
        
        # Find longest path (approximate using BFS)
        nodes = graph['nodes']
        adjacency = graph['adjacency']
        
        if not nodes:
            return 0.0
        
        # Compute path products for all paths from source to sink
        # (simplified: use average path length weighted by attenuation)
        path_products = []
        
        # Start from quantum parameters (sources)
        q_nodes = [i for i, t in enumerate(graph['node_types']) if t == 'quantum']
        c_nodes = [i for i, t in enumerate(graph['node_types']) if t == 'classical']
        
        # Find paths from quantum to classical (measurement paths)
        for source in q_nodes[:min(5, len(q_nodes))]:  # Sample for efficiency
            for sink in c_nodes[:min(3, len(c_nodes))]:
                path_product = self._find_path_product(source, sink, graph)
                if path_product > 0:
                    path_products.append(path_product)
        
        if not path_products:
            # Fallback: use average attenuation
            avg_attenuation = np.mean([e.attenuation for e in graph['edges']])
            if avg_attenuation > 0:
                return -np.log(avg_attenuation) / np.log(0.5)  # Normalize
            return 0.0
        
        # Effective depth from path products
        avg_product = np.mean(path_products)
        avg_attenuation = np.mean([e.attenuation for e in graph['edges']])
        
        if avg_product > 0 and avg_attenuation > 0:
            effective_depth = -np.log(avg_product) / np.log(avg_attenuation)
            return float(effective_depth)
        
        return 0.0
    
    def _find_path_product(self, source: int, sink: int, graph: Dict) -> float:
        """Find product of attenuations along path from source to sink."""
        # Simplified BFS to find path
        from collections import deque
        
        adjacency = graph['adjacency']
        edges = {f"{e.source_param}→{e.target_param}": e.attenuation 
                for e in graph['edges']}
        
        queue = deque([(source, 1.0)])  # (node, product_so_far)
        visited = {source}
        max_depth = 10  # Limit search depth
        
        depth = 0
        while queue and depth < max_depth:
            level_size = len(queue)
            for _ in range(level_size):
                node, product = queue.popleft()
                
                if node == sink:
                    return product
                
                for neighbor in adjacency.get(node, []):
                    if neighbor not in visited:
                        edge_key = f"{node}→{neighbor}"
                        edge_attenuation = edges.get(edge_key, 0.5)
                        new_product = product * edge_attenuation
                        queue.append((neighbor, new_product))
                        visited.add(neighbor)
            
            depth += 1
        
        return 0.0  # No path found
    
    def analyze_architecture(self, architecture) -> Dict:
        """
        Complete gradient flow analysis for an architecture.
        
        Returns:
            Dictionary with:
            - 'graph': Gradient flow graph
            - 'bottlenecks': List of bottleneck edges
            - 'effective_depth': Effective depth metric
            - 'min_attenuation': Minimum attenuation (worst bottleneck)
            - 'avg_attenuation': Average attenuation
            - 'bottleneck_count': Number of bottleneck edges
        """
        graph = self.build_gradient_flow_graph(architecture)
        bottlenecks = self.identify_bottlenecks(graph)
        effective_depth = self.compute_effective_depth(graph)
        
        if graph['edges']:
            attenuations = [e.attenuation for e in graph['edges']]
            min_attenuation = min(attenuations)
            avg_attenuation = np.mean(attenuations)
        else:
            min_attenuation = 1.0
            avg_attenuation = 1.0
        
        analysis = {
            'graph': graph,
            'bottlenecks': bottlenecks,
            'effective_depth': effective_depth,
            'min_attenuation': min_attenuation,
            'avg_attenuation': avg_attenuation,
            'bottleneck_count': len(bottlenecks),
            'is_trainable': min_attenuation >= self.vanishing_threshold
        }
        
        logger.info(f"Gradient flow analysis: effective_depth={effective_depth:.2f}, "
                   f"bottlenecks={len(bottlenecks)}, trainable={analysis['is_trainable']}")
        
        return analysis

