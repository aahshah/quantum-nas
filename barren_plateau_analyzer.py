"""
Barren Plateau Analyzer for Quantum Circuits
Publication-Grade Implementation

Mathematical Foundation:
- Gradient variance: Var(∂_θ⟨O⟩) ~ e^(-αD) where D = depth
- Barren plateau risk increases exponentially with depth
- Local vs global observables affect gradient scaling

Based on:
- McClean et al. "Barren plateaus in quantum neural network training landscapes" (2018)
- Cerezo et al. "Cost-function-dependent barren plateaus in shallow quantum neural networks" (2021)

Author: GQAS Research Team
"""

import torch
import numpy as np
import pennylane as qml
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class BarrenPlateauResult:
    """Result of barren plateau analysis."""
    gradient_variance: float  # Mean gradient variance across parameters
    gradient_variance_per_param: np.ndarray  # Variance for each parameter
    gradient_mean: float  # Mean gradient magnitude
    barren_plateau_risk: float  # Risk score [0, 1] (1 = high risk)
    is_barren: bool  # True if circuit is in barren plateau
    depth_factor: float  # Contribution of depth to barren plateau
    qubit_factor: float  # Contribution of qubit count to barren plateau
    entanglement_factor: float  # Contribution of entanglement to barren plateau
    n_samples: int  # Number of random initializations used


class BarrenPlateauAnalyzer:
    """
    Analyzes quantum circuits for barren plateau susceptibility.
    
    Key Metrics:
    - Gradient variance: Var(∂_θ⟨O⟩) across random initializations
    - Barren plateau risk: Normalized risk score [0, 1]
    - Depth scaling: How gradient variance scales with depth
    
    Theory:
    For parameterized quantum state |ψ(θ)⟩ and observable O:
    
        Var(∂_θ⟨O⟩) ~ e^(-αD)
    
    where D = circuit depth, α = problem-dependent constant.
    
    Circuits with Var < threshold are in barren plateau.
    """
    
    def __init__(self,
                 n_samples: int = 100,
                 variance_threshold: float = 1e-6,
                 use_parameter_shift: bool = True):
        """
        Initialize barren plateau analyzer.
        
        Args:
            n_samples: Number of random parameter initializations
            variance_threshold: Gradient variance below this = barren plateau
            use_parameter_shift: Use parameter-shift rule (exact) vs finite diff
        """
        self.n_samples = n_samples
        self.variance_threshold = variance_threshold
        self.use_parameter_shift = use_parameter_shift
        self.cache = {}  # Cache results for architectures
        
    def analyze_circuit(self,
                       circuit: Callable,
                       n_params: int,
                       observable_type: str = 'local',
                       device = None) -> BarrenPlateauResult:
        """
        Analyze circuit for barren plateau susceptibility.
        
        Algorithm:
        1. Sample n_samples random parameter initializations
        2. For each, compute gradient using parameter-shift rule
        3. Compute variance of gradients across samples
        4. Assess barren plateau risk based on variance
        
        Args:
            circuit: Quantum circuit function
            n_params: Number of parameters
            observable_type: 'local' or 'global' (affects scaling)
            device: PennyLane device
            
        Returns:
            BarrenPlateauResult with analysis metrics
        """
        if circuit is None:
            raise ValueError("Circuit cannot be None")
        if n_params == 0:
            return BarrenPlateauResult(
                gradient_variance=0.0,
                gradient_variance_per_param=np.array([]),
                gradient_mean=0.0,
                barren_plateau_risk=1.0,
                is_barren=True,
                depth_factor=0.0,
                qubit_factor=0.0,
                entanglement_factor=0.0,
                n_samples=0
            )
        
        # Check cache
        cache_key = f"bp_{n_params}_{observable_type}_{self.n_samples}"
        if cache_key in self.cache:
            logger.debug("Using cached barren plateau result")
            return self.cache[cache_key]
        
        try:
            # Get device if not provided
            if device is None:
                if hasattr(circuit, 'device'):
                    device = circuit.device
                else:
                    # Estimate qubits from circuit
                    n_qubits = self._estimate_qubits(circuit, n_params)
                    device = qml.device('default.qubit', wires=n_qubits)
            
            # Compute gradients for multiple random initializations
            gradients = []
            
            for _ in range(self.n_samples):
                # Random parameter initialization
                params = np.random.randn(n_params) * 0.1
                
                try:
                    if self.use_parameter_shift:
                        # Use parameter-shift rule (exact gradients)
                        grad = self._compute_gradient_parameter_shift(
                            circuit, params, device
                        )
                    else:
                        # Finite difference approximation
                        grad = self._compute_gradient_finite_diff(
                            circuit, params, device
                        )
                    
                    if grad is not None and len(grad) > 0:
                        gradients.append(grad)
                except Exception as e:
                    logger.debug(f"Error computing gradient: {e}")
                    continue
            
            if not gradients:
                # Fallback: estimate based on circuit properties
                logger.warning("No gradients computed, using estimation")
                return self._estimate_barren_plateau(n_params, observable_type)
            
            gradients = np.array(gradients)
            
            # Compute statistics
            gradient_variance_per_param = np.var(gradients, axis=0)
            gradient_variance = float(np.mean(gradient_variance_per_param))
            gradient_mean = float(np.mean(np.abs(gradients)))
            
            # Assess barren plateau risk
            # Risk increases as variance decreases
            # Normalize: risk = 1 - min(1, variance / threshold)
            if gradient_variance < self.variance_threshold:
                barren_plateau_risk = 1.0
                is_barren = True
            else:
                # Risk scales inversely with variance
                risk_factor = self.variance_threshold / (gradient_variance + 1e-10)
                barren_plateau_risk = float(np.clip(risk_factor, 0.0, 1.0))
                is_barren = barren_plateau_risk > 0.5
            
            # Estimate contributing factors (for analysis)
            depth_factor = self._estimate_depth_factor(circuit, n_params)
            qubit_factor = self._estimate_qubit_factor(circuit, n_params)
            entanglement_factor = self._estimate_entanglement_factor(circuit)
            
            result = BarrenPlateauResult(
                gradient_variance=gradient_variance,
                gradient_variance_per_param=gradient_variance_per_param,
                gradient_mean=gradient_mean,
                barren_plateau_risk=barren_plateau_risk,
                is_barren=is_barren,
                depth_factor=depth_factor,
                qubit_factor=qubit_factor,
                entanglement_factor=entanglement_factor,
                n_samples=len(gradients)
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            logger.info(f"Barren plateau analysis: variance={gradient_variance:.6f}, "
                       f"risk={barren_plateau_risk:.3f}, is_barren={is_barren}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in barren plateau analysis: {e}")
            # Return conservative estimate (high risk)
            return BarrenPlateauResult(
                gradient_variance=1e-8,
                gradient_variance_per_param=np.zeros(n_params),
                gradient_mean=0.0,
                barren_plateau_risk=1.0,
                is_barren=True,
                depth_factor=0.0,
                qubit_factor=0.0,
                entanglement_factor=0.0,
                n_samples=0
            )
    
    def _compute_gradient_parameter_shift(self,
                                         circuit: Callable,
                                         params: np.ndarray,
                                         device) -> np.ndarray:
        """
        Compute gradient using parameter-shift rule.
        
        For parameter θᵢ:
            ∂⟨O⟩/∂θᵢ = (⟨O(θ + π/2·eᵢ)⟩ - ⟨O(θ - π/2·eᵢ)⟩) / 2
        
        This gives exact gradients for parameterized quantum circuits.
        """
        try:
            # Create gradient function using PennyLane
            if hasattr(circuit, '__call__'):
                # Try to use PennyLane's grad function
                try:
                    grad_fn = qml.grad(circuit)
                    grad = grad_fn(params)
                    if isinstance(grad, (list, tuple)):
                        grad = np.array(grad)
                    return grad
                except:
                    # Fallback: manual parameter-shift
                    return self._manual_parameter_shift(circuit, params)
            else:
                return self._manual_parameter_shift(circuit, params)
        except Exception as e:
            logger.debug(f"Parameter-shift failed: {e}")
            return None
    
    def _manual_parameter_shift(self,
                                circuit: Callable,
                                params: np.ndarray) -> np.ndarray:
        """Manual implementation of parameter-shift rule."""
        gradients = []
        shift_angle = np.pi / 2
        
        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += shift_angle
            try:
                output_plus = circuit(params_plus)
                if isinstance(output_plus, (list, tuple)):
                    output_plus = output_plus[0] if len(output_plus) > 0 else 0.0
                output_plus = float(output_plus)
            except:
                output_plus = 0.0
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= shift_angle
            try:
                output_minus = circuit(params_minus)
                if isinstance(output_minus, (list, tuple)):
                    output_minus = output_minus[0] if len(output_minus) > 0 else 0.0
                output_minus = float(output_minus)
            except:
                output_minus = 0.0
            
            # Gradient: (f(θ + π/2) - f(θ - π/2)) / 2
            grad_i = (output_plus - output_minus) / 2.0
            gradients.append(grad_i)
        
        return np.array(gradients)
    
    def _compute_gradient_finite_diff(self,
                                     circuit: Callable,
                                     params: np.ndarray,
                                     device,
                                     epsilon: float = 1e-5) -> np.ndarray:
        """Compute gradient using finite differences (backup method)."""
        gradients = []
        base_output = circuit(params)
        if isinstance(base_output, (list, tuple)):
            base_output = base_output[0] if len(base_output) > 0 else 0.0
        base_output = float(base_output)
        
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += epsilon
            
            try:
                perturbed_output = circuit(params_perturbed)
                if isinstance(perturbed_output, (list, tuple)):
                    perturbed_output = perturbed_output[0] if len(perturbed_output) > 0 else 0.0
                perturbed_output = float(perturbed_output)
                
                grad_i = (perturbed_output - base_output) / epsilon
                gradients.append(grad_i)
            except:
                gradients.append(0.0)
        
        return np.array(gradients)
    
    def _estimate_qubits(self, circuit: Callable, n_params: int) -> int:
        """Estimate number of qubits from circuit."""
        if hasattr(circuit, 'device'):
            if hasattr(circuit.device, 'num_wires'):
                return circuit.device.num_wires
        
        # Heuristic: ~3 params per qubit per layer
        return max(2, n_params // 3)
    
    def _estimate_depth_factor(self, circuit: Callable, n_params: int) -> float:
        """Estimate how depth contributes to barren plateau."""
        # Deeper circuits = higher risk
        # Estimate depth from parameter count (rough heuristic)
        estimated_depth = max(1, n_params // 4)
        # Exponential scaling: e^(-αD)
        depth_factor = np.exp(-0.1 * estimated_depth)
        return float(depth_factor)
    
    def _estimate_qubit_factor(self, circuit: Callable, n_params: int) -> float:
        """Estimate how qubit count contributes to barren plateau."""
        n_qubits = self._estimate_qubits(circuit, n_params)
        # More qubits = higher risk (exponential in worst case)
        qubit_factor = np.exp(-0.05 * n_qubits)
        return float(qubit_factor)
    
    def _estimate_entanglement_factor(self, circuit: Callable) -> float:
        """Estimate how entanglement contributes to barren plateau."""
        # High entanglement = higher risk
        # Default: moderate entanglement
        return 0.5
    
    def _estimate_barren_plateau(self, n_params: int, observable_type: str) -> BarrenPlateauResult:
        """Estimate barren plateau metrics when exact computation fails."""
        # Conservative estimate based on parameter count
        # More parameters = higher risk of barren plateau
        risk = min(1.0, 1.0 - np.exp(-0.01 * n_params))
        
        return BarrenPlateauResult(
            gradient_variance=1e-8,
            gradient_variance_per_param=np.zeros(n_params),
            gradient_mean=0.0,
            barren_plateau_risk=risk,
            is_barren=risk > 0.5,
            depth_factor=0.5,
            qubit_factor=0.5,
            entanglement_factor=0.5,
            n_samples=0
        )
    
    def analyze_architecture(self,
                           architecture,
                           quantum_simulator,
                           X_sample: np.ndarray,
                           observable_type: str = 'local') -> BarrenPlateauResult:
        """
        Analyze architecture for barren plateau susceptibility.
        
        This is a convenience wrapper that builds the circuit and analyzes it.
        
        Args:
            architecture: HybridArchitecture object
            quantum_simulator: QuantumCircuitSimulator instance
            X_sample: Sample input data
            observable_type: 'local' or 'global' observable
            
        Returns:
            BarrenPlateauResult
        """
        if architecture is None or not architecture.quantum_layers:
            return BarrenPlateauResult(
                gradient_variance=0.0,
                gradient_variance_per_param=np.array([]),
                gradient_mean=0.0,
                barren_plateau_risk=1.0,
                is_barren=True,
                depth_factor=0.0,
                qubit_factor=0.0,
                entanglement_factor=0.0,
                n_samples=0
            )
        
        try:
            # Build circuit
            circuit, n_params, n_qubits = quantum_simulator.build_circuit(
                architecture,
                X_sample.shape[1] if len(X_sample.shape) > 1 else len(X_sample)
            )
            
            if circuit is None or n_params == 0:
                return self._estimate_barren_plateau(0, observable_type)
            
            # Analyze circuit
            result = self.analyze_circuit(
                circuit, n_params, observable_type
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Error analyzing architecture: {e}")
            # Return estimate based on architecture properties
            n_params = architecture.trainable_params
            return self._estimate_barren_plateau(n_params, observable_type)
    
    def compare_architectures(self,
                            architectures: List,
                            quantum_simulator,
                            X_sample: np.ndarray) -> Dict:
        """
        Compare multiple architectures for barren plateau susceptibility.
        
        Useful for comparing Depth-2 vs Depth-6 circuits.
        
        Returns:
            Dictionary with comparison metrics
        """
        results = {}
        
        for i, arch in enumerate(architectures):
            try:
                result = self.analyze_architecture(
                    arch, quantum_simulator, X_sample
                )
                results[f'arch_{i}'] = {
                    'depth': arch.total_depth,
                    'n_qubits': arch.total_qubits,
                    'gradient_variance': result.gradient_variance,
                    'barren_plateau_risk': result.barren_plateau_risk,
                    'is_barren': result.is_barren
                }
            except Exception as e:
                logger.warning(f"Error analyzing architecture {i}: {e}")
                results[f'arch_{i}'] = {
                    'depth': arch.total_depth,
                    'n_qubits': arch.total_qubits,
                    'gradient_variance': 0.0,
                    'barren_plateau_risk': 1.0,
                    'is_barren': True
                }
        
        return results
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.cache.clear()
        logger.debug("Barren plateau cache cleared")


def compute_gradient_variance_for_architecture(architecture,
                                               quantum_simulator,
                                               X_sample: np.ndarray,
                                               n_samples: int = 50) -> float:
    """
    Convenience function to compute gradient variance for an architecture.
    
    Returns normalized gradient variance [0, 1] for use in MO-QAS.
    """
    analyzer = BarrenPlateauAnalyzer(n_samples=n_samples)
    result = analyzer.analyze_architecture(architecture, quantum_simulator, X_sample)
    
    # Normalize gradient variance to [0, 1]
    # Typical values: 1e-8 (barren) to 1e-2 (trainable)
    # Use log scale for normalization
    if result.gradient_variance > 0:
        log_var = np.log10(result.gradient_variance + 1e-10)
        # Map from [-8, -2] to [0, 1]
        normalized = (log_var + 8) / 6.0
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = 0.0
    
    return float(normalized)

