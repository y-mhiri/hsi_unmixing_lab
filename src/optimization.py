"""
Optimization methods for hyperspectral unmixing.
"""

import numpy as np
from typing import Tuple, List
import warnings


class HyperspectralUnmixer:
    """
    A class implementing various optimization methods for hyperspectral unmixing.
    """
    
    def __init__(self):
        """Initialize the unmixer."""
        pass
    
    def unconstrained_least_squares(self, S: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Solve unconstrained least squares problem: min ||SA - Y||_F^2
        
        Parameters:
        -----------
        S : np.ndarray
            Endmember matrix with shape (bands, num_endmembers)
        Y : np.ndarray
            Data matrix with shape (bands, num_pixels)
            
        Returns:
        --------
        np.ndarray
            Abundance matrix with shape (num_endmembers, num_pixels)
        """
        # A* = (S^T S)^(-1) S^T Y
        StS = S.T @ S
        StY = S.T @ Y
        
        # Check if S^T S is invertible
        try:
            A = np.linalg.solve(StS, StY)
        except np.linalg.LinAlgError:
            warnings.warn("S^T S is singular, using pseudo-inverse")
            A = np.linalg.pinv(S) @ Y
            
        return A
    
    def sum_to_one_constrained_ls(self, S: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Solve sum-to-one constrained least squares problem.
        
        min ||SA - Y||_F^2  subject to  1^T A = 1^T
        
        Parameters:
        -----------
        S : np.ndarray
            Endmember matrix with shape (bands, num_endmembers)
        Y : np.ndarray
            Data matrix with shape (bands, num_pixels)
            
        Returns:
        --------
        np.ndarray
            Abundance matrix with shape (num_endmembers, num_pixels)
        """
        num_endmembers, num_pixels = S.shape[1], Y.shape[1]
        
        # Precompute matrices
        StS = S.T @ S
        StY = S.T @ Y
        ones = np.ones((num_endmembers, 1))
        
        # Compute the constrained solution
        # A = (S^T S)^(-1) S^T Y + (S^T S)^(-1) 1 * lambda
        # where lambda is chosen to satisfy the constraint
        
        try:
            StS_inv = np.linalg.inv(StS)
        except np.linalg.LinAlgError:
            warnings.warn("S^T S is singular, using pseudo-inverse")
            StS_inv = np.linalg.pinv(StS)
        
        # Unconstrained solution
        A_unconstrained = StS_inv @ StY
        
        # Compute lambda for each pixel to satisfy sum-to-one constraint
        # lambda = (1 - 1^T A_unconstrained) / (1^T (S^T S)^(-1) 1)
        denominator = ones.T @ StS_inv @ ones
        numerator = 1 - ones.T @ A_unconstrained
        
        lambda_values = numerator / denominator
        
        # Final constrained solution
        A = A_unconstrained + StS_inv @ ones @ lambda_values
        
        return A
    
    def project_onto_simplex(self, v: np.ndarray) -> np.ndarray:
        """
        Project a vector onto the unit simplex using Duchi et al. algorithm.
        
        Parameters:
        -----------
        v : np.ndarray
            Input vector
            
        Returns:
        --------
        np.ndarray
            Projected vector on the simplex
        """
        n = len(v)
        
        # Sort in descending order
        u = np.sort(v)[::-1]
        
        # Find rho
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(n) + 1
        cond = u - cssv / ind > 0
        
        if np.any(cond):
            rho = np.where(cond)[0][-1]
            theta = cssv[rho] / (rho + 1.0)
        else:
            theta = 0.0
        
        # Project
        w = np.maximum(v - theta, 0)
        return w
    
    def projected_gradient_descent(self, S: np.ndarray, Y: np.ndarray, 
                                  constraint_type: str = 'non_negative',
                                  max_iter: int = 1000, 
                                  step_size: float = .001,
                                  tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Solve constrained least squares using projected gradient descent.
        
        Parameters:
        -----------
        S : np.ndarray
            Endmember matrix with shape (bands, num_endmembers)
        Y : np.ndarray
            Data matrix with shape (bands, num_pixels)
        constraint_type : str
            Type of constraint: 'non_negative', 'simplex', or 'sum_to_one'
        max_iter : int
            Maximum number of iterations
        step_size : float
            Step size for gradient descent
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        Tuple[np.ndarray, List[float]]
            Abundance matrix and list of objective values
        """
        num_endmembers, num_pixels = S.shape[1], Y.shape[1]


        # Initialize abundances with unconstrained
        A = self.unconstrained_least_squares(S,Y)
        if constraint_type == 'non_negative':
            A = np.maximum(A, 0)
        elif constraint_type == 'sum_to_one':
            # Project each pixel to sum to 1
            A = A / np.sum(A, axis=0, keepdims=True)
        elif constraint_type == 'simplex':
            # Project each pixel onto the simplex
            for p in range(num_pixels):
                A[:, p] = self.project_onto_simplex(A[:, p])

        objective_values = []
        StS = S.T @ S
        StY = S.T @ Y
        
        for iteration in range(max_iter):
            # Compute objective function
            residual = S @ A - Y
            objective = 0.5 * np.sum(residual**2)
            objective_values.append(objective)
            
            # Compute gradient: grad_A = S^T (SA - Y)
            gradient = StS @ A - StY
            
            # Gradient descent step
            A_new = A - step_size * gradient
            
            # Apply constraints
            if constraint_type == 'non_negative':
                A_new = np.maximum(A_new, 0)
            elif constraint_type == 'sum_to_one':
                # Project each pixel to sum to 1
                A_new = A_new / np.sum(A_new, axis=0, keepdims=True)
            elif constraint_type == 'simplex':
                # Project each pixel onto the simplex
                for p in range(num_pixels):
                    A_new[:, p] = self.project_onto_simplex(A_new[:, p])
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(objective_values[-1] - objective_values[-2]) / objective_values[-2]
                if rel_change < tolerance:
                    print(f"Converged after {iteration+1} iterations")
                    break
            
            A = A_new
        
        return A, objective_values
    
    def fully_constrained_least_squares(self, S: np.ndarray, Y: np.ndarray,
                                       max_iter: int = 1000,
                                       step_size: float = 0.001,
                                       tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Solve fully constrained least squares (simplex constraints).
        
        min ||SA - Y||_F^2  subject to  A >= 0, 1^T A = 1^T
        
        Parameters:
        -----------
        S : np.ndarray
            Endmember matrix with shape (bands, num_endmembers)
        Y : np.ndarray
            Data matrix with shape (bands, num_pixels)
        max_iter : int
            Maximum number of iterations
        step_size : float
            Step size for gradient descent
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        Tuple[np.ndarray, List[float]]
            Abundance matrix and list of objective values
        """
        return self.projected_gradient_descent(S, Y, constraint_type='simplex',
                                             max_iter=max_iter, step_size=step_size,
                                             tolerance=tolerance)


class BlindUnmixer:
    """
    Implementation of blind hyperspectral unmixing using Block Coordinate Descent.
    """
    
    def __init__(self, unmixer: HyperspectralUnmixer):
        """
        Initialize blind unmixer.
        
        Parameters:
        -----------
        unmixer : HyperspectralUnmixer
            Instance of constrained unmixer for subproblems
        """
        self.unmixer = unmixer
    
    def block_coordinate_descent(self, Y: np.ndarray, num_endmembers: int,
                                max_iter: int = 100,
                                inner_max_iter: int = 500,
                                tolerance: float = 1e-6,
                                initialization: str = 'random') -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Solve blind unmixing using Block Coordinate Descent.
        
        min ||SA - Y||_F^2  subject to  A >= 0, 1^T A = 1^T, S >= 0
        
        Parameters:
        -----------
        Y : np.ndarray
            Data matrix with shape (bands, num_pixels)
        num_endmembers : int
            Number of endmembers to estimate
        max_iter : int
            Maximum number of outer iterations
        inner_max_iter : int
            Maximum iterations for inner subproblems
        tolerance : float
            Convergence tolerance
        initialization : str
            Initialization strategy: 'random' or 'vca'
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, List[float]]
            Estimated endmembers S, abundances A, and objective values
        """
        bands, num_pixels = Y.shape
        
        # Initialize endmembers and abundances
        if initialization == 'random':
            S = np.random.rand(bands, num_endmembers)
            S = S / np.linalg.norm(S, axis=0)  # Normalize columns
            A = np.random.dirichlet(np.ones(num_endmembers), num_pixels).T
        else:
            # Could implement VCA or other initialization methods
            S = np.random.rand(bands, num_endmembers)
            S = S / np.linalg.norm(S, axis=0)
            A = np.random.dirichlet(np.ones(num_endmembers), num_pixels).T
        
        objective_values = []
        
        for iteration in range(max_iter):
            # Compute current objective
            residual = S @ A - Y
            objective = 0.5 * np.sum(residual**2)
            objective_values.append(objective)
            
            # Update A with S fixed (simplex-constrained least squares)
            A, _ = self.unmixer.fully_constrained_least_squares(
                S, Y, max_iter=inner_max_iter, tolerance=tolerance*0.1)
            
            # Update S with A fixed (non-negative least squares)
            # Solve: min ||SA - Y||_F^2 subject to S >= 0
            # This is equivalent to: min ||A^T S^T - Y^T||_F^2 subject to S^T >= 0
            S_new, _ = self.unmixer.projected_gradient_descent(
                A.T, Y.T, constraint_type='non_negative', 
                max_iter=inner_max_iter, tolerance=tolerance*0.1)
            S = S_new.T
            
            # Normalize endmembers (optional, helps with scaling)
            S = S / np.linalg.norm(S, axis=0)
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(objective_values[-1] - objective_values[-2]) / objective_values[-2]
                if rel_change < tolerance:
                    print(f"BCD converged after {iteration+1} iterations")
                    break
        
        return S, A, objective_values


def adaptive_step_size(S: np.ndarray, initial_step: float = 0.01) -> float:
    """
    Compute adaptive step size based on spectral properties of S.
    
    Parameters:
    -----------
    S : np.ndarray
        Endmember matrix
    initial_step : float
        Initial step size
        
    Returns:
    --------
    float
        Adaptive step size
    """
    # Use reciprocal of largest eigenvalue of S^T S for stability
    StS = S.T @ S
    max_eigenvalue = np.max(np.linalg.eigvals(StS))
    adaptive_step = min(initial_step, 0.9 / max_eigenvalue)
    return adaptive_step


def check_algorithm_convergence(objective_values: List[float], 
                               tolerance: float = 1e-6,
                               min_iterations: int = 10) -> bool:
    """
    Check if optimization algorithm has converged.
    
    Parameters:
    -----------
    objective_values : List[float]
        List of objective function values
    tolerance : float
        Convergence tolerance
    min_iterations : int
        Minimum number of iterations before checking convergence
        
    Returns:
    --------
    bool
        True if converged
    """
    if len(objective_values) < min_iterations:
        return False
    
    # Check relative change in objective
    recent_change = abs(objective_values[-1] - objective_values[-2]) / objective_values[-2]
    return recent_change < tolerance