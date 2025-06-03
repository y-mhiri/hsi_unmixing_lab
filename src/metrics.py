"""
Evaluation metrics for hyperspectral unmixing performance assessment.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
import warnings


class UnmixingEvaluator:
    """
    A class for evaluating hyperspectral unmixing performance.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def spectral_angle_mapper(self, true_spectrum: np.ndarray, 
                             reconstructed_spectrum: np.ndarray) -> float:
        """
        Compute Spectral Angle Mapper (SAM) between two spectra.
        
        SAM measures the angle between two spectral vectors, providing
        a measure of spectral similarity independent of illumination.
        
        Parameters:
        -----------
        true_spectrum : np.ndarray
            Reference spectrum
        reconstructed_spectrum : np.ndarray
            Reconstructed spectrum
            
        Returns:
        --------
        float
            SAM value in radians (0 = perfect match)
        """
        # Normalize vectors
        true_norm = np.linalg.norm(true_spectrum)
        recon_norm = np.linalg.norm(reconstructed_spectrum)
        
        if true_norm == 0 or recon_norm == 0:
            return np.pi / 2  # Maximum angle for zero vectors
        
        # Compute cosine of angle
        cos_angle = np.dot(true_spectrum, reconstructed_spectrum) / (true_norm * recon_norm)
        
        # Clip to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Return angle in radians
        return np.arccos(cos_angle)
    
    def root_mean_square_error(self, true_data: np.ndarray, 
                              reconstructed_data: np.ndarray) -> float:
        """
        Compute Root Mean Square Error (RMSE) between true and reconstructed data.
        
        Parameters:
        -----------
        true_data : np.ndarray
            Ground truth data
        reconstructed_data : np.ndarray
            Reconstructed data
            
        Returns:
        --------
        float
            RMSE value
        """
        mse = np.mean((true_data - reconstructed_data)**2)
        return np.sqrt(mse)
    
    def mean_absolute_error(self, true_data: np.ndarray, 
                           reconstructed_data: np.ndarray) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Parameters:
        -----------
        true_data : np.ndarray
            Ground truth data
        reconstructed_data : np.ndarray
            Reconstructed data
            
        Returns:
        --------
        float
            MAE value
        """
        return np.mean(np.abs(true_data - reconstructed_data))
    
    def signal_to_noise_ratio(self, true_data: np.ndarray, 
                             reconstructed_data: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR) in dB.
        
        Parameters:
        -----------
        true_data : np.ndarray
            Ground truth data
        reconstructed_data : np.ndarray
            Reconstructed data
            
        Returns:
        --------
        float
            SNR in dB
        """
        signal_power = np.mean(true_data**2)
        noise_power = np.mean((true_data - reconstructed_data)**2)
        
        if noise_power == 0:
            return np.inf
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def structural_similarity_index(self, true_image: np.ndarray, 
                                   reconstructed_image: np.ndarray,
                                   data_range: Optional[float] = None) -> float:
        """
        Compute Structural Similarity Index (SSIM) for 2D images.
        
        Parameters:
        -----------
        true_image : np.ndarray
            Ground truth image
        reconstructed_image : np.ndarray
            Reconstructed image
        data_range : Optional[float]
            Data range of the images
            
        Returns:
        --------
        float
            SSIM value (between -1 and 1, 1 = perfect match)
        """
        if data_range is None:
            data_range = np.max([true_image.max(), reconstructed_image.max()]) - \
                        np.min([true_image.min(), reconstructed_image.min()])
        
        return ssim(true_image, reconstructed_image, data_range=data_range)
    
    def abundance_angle_distance(self, true_abundances: np.ndarray, 
                                estimated_abundances: np.ndarray) -> float:
        """
        Compute Abundance Angle Distance (AAD) between abundance vectors.
        
        Parameters:
        -----------
        true_abundances : np.ndarray
            True abundance matrix (num_endmembers, num_pixels)
        estimated_abundances : np.ndarray
            Estimated abundance matrix (num_endmembers, num_pixels)
            
        Returns:
        --------
        float
            Mean AAD across all pixels
        """
        num_pixels = true_abundances.shape[1]
        aad_values = []
        
        for p in range(num_pixels):
            aad = self.spectral_angle_mapper(true_abundances[:, p], 
                                           estimated_abundances[:, p])
            aad_values.append(aad)
        
        return np.mean(aad_values)
    
    def evaluate_reconstruction(self, S: np.ndarray, A: np.ndarray, 
                               Y_true: np.ndarray, shape: Tuple[int, int],
                               rgb_bands: Optional[Tuple[int, int, int]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of hyperspectral reconstruction.
        
        Parameters:
        -----------
        S : np.ndarray
            Endmember matrix (bands, num_endmembers)
        A : np.ndarray
            Abundance matrix (num_endmembers, num_pixels)
        Y_true : np.ndarray
            True data matrix (bands, num_pixels)
        shape : Tuple[int, int]
            Spatial dimensions (height, width)
        rgb_bands : Optional[Tuple[int, int, int]]
            RGB band indices for SSIM computation
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        # Reconstruct data
        Y_reconstructed = S @ A
        
        # Spectral metrics
        sam_values = []
        num_pixels = Y_true.shape[1]
        
        for p in range(num_pixels):
            sam = self.spectral_angle_mapper(Y_true[:, p], Y_reconstructed[:, p])
            sam_values.append(sam)
        
        mean_sam = np.mean(sam_values)
        std_sam = np.std(sam_values)
        
        # Global reconstruction metrics
        rmse = self.root_mean_square_error(Y_true, Y_reconstructed)
        mae = self.mean_absolute_error(Y_true, Y_reconstructed)
        snr = self.signal_to_noise_ratio(Y_true, Y_reconstructed)
        
        results = {
            'mean_sam_radians': mean_sam,
            'mean_sam_degrees': np.degrees(mean_sam),
            'std_sam_degrees': np.degrees(std_sam),
            'rmse': rmse,
            'mae': mae,
            'snr_db': snr
        }
        
        # Compute SSIM for RGB composite if bands are provided
        if rgb_bands is not None:
            height, width = shape
            
            # Reshape to image format
            Y_true_img = Y_true.T.reshape(height, width, -1)
            Y_recon_img = Y_reconstructed.T.reshape(height, width, -1)
            
            # Extract RGB bands
            red_idx, green_idx, blue_idx = rgb_bands
            
            try:
                ssim_red = self.structural_similarity_index(
                    Y_true_img[:, :, red_idx], Y_recon_img[:, :, red_idx])
                ssim_green = self.structural_similarity_index(
                    Y_true_img[:, :, green_idx], Y_recon_img[:, :, green_idx])
                ssim_blue = self.structural_similarity_index(
                    Y_true_img[:, :, blue_idx], Y_recon_img[:, :, blue_idx])
                
                results.update({
                    'ssim_red': ssim_red,
                    'ssim_green': ssim_green,
                    'ssim_blue': ssim_blue,
                    'ssim_mean': np.mean([ssim_red, ssim_green, ssim_blue])
                })
            except Exception as e:
                warnings.warn(f"Could not compute SSIM: {e}")
        
        return results
    
    def evaluate_abundances(self, A_estimated: np.ndarray) -> Dict[str, float]:
        """
        Evaluate physical constraints of estimated abundances.
        
        Parameters:
        -----------
        A_estimated : np.ndarray
            Estimated abundance matrix (num_endmembers, num_pixels)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing abundance constraint metrics
        """
        num_endmembers, num_pixels = A_estimated.shape
        
        # Check non-negativity constraint
        negative_values = np.sum(A_estimated < 0)
        fraction_negative = negative_values / (num_endmembers * num_pixels)
        
        # Check sum-to-one constraint
        abundance_sums = np.sum(A_estimated, axis=0)
        sum_deviation = np.abs(abundance_sums - 1.0)
        mean_sum_deviation = np.mean(sum_deviation)
        max_sum_deviation = np.max(sum_deviation)
        
        # Abundance statistics
        mean_abundance = np.mean(A_estimated)
        std_abundance = np.std(A_estimated)
        min_abundance = np.min(A_estimated)
        max_abundance = np.max(A_estimated)
        
        return {
            'fraction_negative': fraction_negative,
            'mean_sum_deviation': mean_sum_deviation,
            'max_sum_deviation': max_sum_deviation,
            'mean_abundance': mean_abundance,
            'std_abundance': std_abundance,
            'min_abundance': min_abundance,
            'max_abundance': max_abundance,
            'fraction_in_simplex': np.mean((A_estimated >= 0).all(axis=0) & 
                                         (np.abs(abundance_sums - 1.0) < 0.01))
        }
    
    def compare_methods(self, results_dict: Dict[str, Dict[str, float]]) -> None:
        """
        Compare multiple unmixing methods.
        
        Parameters:
        -----------
        results_dict : Dict[str, Dict[str, float]]
            Dictionary mapping method names to their evaluation results
        """
        import pandas as pd
        
        # Convert to DataFrame for easy comparison
        df = pd.DataFrame(results_dict).T
        
        print("Comparison of Unmixing Methods:")
        print("=" * 50)
        print(df.round(4))
        
        # Highlight best performance for key metrics
        if 'mean_sam_degrees' in df.columns:
            best_sam = df['mean_sam_degrees'].idxmin()
            print(f"\nBest SAM: {best_sam} ({df.loc[best_sam, 'mean_sam_degrees']:.4f}°)")
        
        if 'rmse' in df.columns:
            best_rmse = df['rmse'].idxmin()
            print(f"Best RMSE: {best_rmse} ({df.loc[best_rmse, 'rmse']:.6f})")
        
        if 'ssim_mean' in df.columns:
            best_ssim = df['ssim_mean'].idxmax()
            print(f"Best SSIM: {best_ssim} ({df.loc[best_ssim, 'ssim_mean']:.4f})")


def compute_endmember_similarity(S_true: np.ndarray, S_estimated: np.ndarray) -> Dict[str, float]:
    """
    Compute similarity between true and estimated endmembers.
    
    Parameters:
    -----------
    S_true : np.ndarray
        True endmember matrix (bands, num_endmembers)
    S_estimated : np.ndarray
        Estimated endmember matrix (bands, num_endmembers)
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing endmember similarity metrics
    """
    evaluator = UnmixingEvaluator()
    
    num_endmembers = S_true.shape[1]
    sam_values = []
    
    # Note: This assumes the order of endmembers is preserved
    # In practice, you might need to solve an assignment problem
    for k in range(min(num_endmembers, S_estimated.shape[1])):
        sam = evaluator.spectral_angle_mapper(S_true[:, k], S_estimated[:, k])
        sam_values.append(sam)
    
    mean_endmember_sam = np.mean(sam_values)
    
    # Compute correlation between endmember matrices
    correlation = np.corrcoef(S_true.flatten(), S_estimated.flatten())[0, 1]
    
    return {
        'mean_endmember_sam_degrees': np.degrees(mean_endmember_sam),
        'endmember_correlation': correlation,
        'individual_sam_degrees': [np.degrees(sam) for sam in sam_values]
    }