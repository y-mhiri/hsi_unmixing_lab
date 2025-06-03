"""
Data loading and preprocessing utilities for hyperspectral images.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Tuple, Optional, Dict, Any
import os


class HyperspectralDataLoader:
    """
    A class to handle loading and preprocessing of hyperspectral datasets,
    specifically designed for the Indian Pines dataset.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the hyperspectral data files
        """
        self.data_path = data_path
        self.hsi_data = None
        self.ground_truth = None
        self.class_names = None

    ## To complete ---------------------
    def load_synthetic(self, height: int = 100, width: int = 100, bands: int = 150, 
                    num_endmembers: int = 5, noise_level: float = 0.05, seed: int = 43, endmembers: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load synthetic hyperspectral dataset with geometric shapes.
        
        Parameters:
        -----------
        height, width : int
            Image dimensions
        bands : int
            Number of spectral bands
        num_endmembers : int
            Number of endmembers
        noise_level : float
            Noise level to add
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            HSI data (H x W x B) and ground truth (H x W)
        """
        from synthetic_image_creator import SyntheticImageCreator
        
        # Create synthetic endmembers
        np.random.seed(seed)
        wavelengths = np.linspace(400, 2500, bands)

        if  endmembers is None:
            S = np.zeros((bands, num_endmembers))
            
            for i in range(num_endmembers):
                center = np.random.uniform(500, 2000)
                width_param = np.random.uniform(200, 500)
                amplitude = np.random.uniform(0.3, 0.8)
                S[:, i] = amplitude * np.exp(-(wavelengths - center)**2 / (2 * width_param**2)) + 0.1
        else: 
            S = endmembers
            num_endmembers = S.shape[1]

        
        # Create synthetic image with shapes
        creator = SyntheticImageCreator(height, width)
        hsi_true, abundance_maps, self.hsi_data = creator.create_synthetic_image(
            S, num_shapes=num_endmembers+2, noise_level=noise_level)
        
        # Create ground truth from abundance maps
        self.ground_truth = np.argmax(abundance_maps, axis=0).astype(np.int32)
        
        # Set class names
        self.class_names = [f'Shape_{i+1}' for i in range(num_endmembers)]
        
        print(f"Created synthetic HSI data with shape: {self.hsi_data.shape}")
        print(f"Created ground truth with shape: {self.ground_truth.shape}")
        print(f"Number of spectral bands: {self.hsi_data.shape[2]}")
        print(f"Number of shapes: {len(np.unique(self.ground_truth))}")
        
        return hsi_true, self.hsi_data, self.ground_truth, abundance_maps
    # ---------------------------------
    def load_pavia(self, 
                   hsi_filename: str = "PaviaU.mat",
                   gt_filename: str = "PaviaU_gt.mat") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the Indian Pines hyperspectral dataset.
        
        Parameters:
        -----------
        hsi_filename : str
            Filename of the hyperspectral data
        gt_filename : str
            Filename of the ground truth data
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            HSI data (H x W x B) and ground truth (H x W)
        """
        # Load hyperspectral data
        hsi_path = os.path.join(self.data_path, hsi_filename)
        hsi_mat = loadmat(hsi_path)
        self.hsi_data = hsi_mat['paviaU'].astype(np.float32)
        
        # Load ground truth
        gt_path = os.path.join(self.data_path, gt_filename)
        gt_mat = loadmat(gt_path)
        self.ground_truth = gt_mat['paviaU_gt'].astype(np.int32)

        # Define class names for Indian Pines
        self.class_names = [
            'Asphalt', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil',
            'Bitumen', 'Self-Blocking Bricks', 'Shadows']

        print(f"Loaded HSI data with shape: {self.hsi_data.shape}")
        print(f"Loaded ground truth with shape: {self.ground_truth.shape}")
        print(f"Number of spectral bands: {self.hsi_data.shape[2]}")
        print(f"Number of classes: {len(np.unique(self.ground_truth))}")
        
        return self.hsi_data, self.ground_truth

    def load_indian_pines(self, 
                         hsi_filename: str = "Indian_pines_corrected.mat",
                         gt_filename: str = "Indian_pines_gt.mat") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the Indian Pines hyperspectral dataset.
        
        Parameters:
        -----------
        hsi_filename : str
            Filename of the hyperspectral data
        gt_filename : str
            Filename of the ground truth data
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            HSI data (H x W x B) and ground truth (H x W)
        """
        # Load hyperspectral data
        hsi_path = os.path.join(self.data_path, hsi_filename)
        hsi_mat = loadmat(hsi_path)
        self.hsi_data = hsi_mat['indian_pines_corrected'].astype(np.float32)
        
        # Load ground truth
        gt_path = os.path.join(self.data_path, gt_filename)
        gt_mat = loadmat(gt_path)
        self.ground_truth = gt_mat['indian_pines_gt'].astype(np.int32)
        
        # Define class names for Indian Pines
        self.class_names = [
            'Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
            'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed',
            'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean',
            'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
        ]
        
        print(f"Loaded HSI data with shape: {self.hsi_data.shape}")
        print(f"Loaded ground truth with shape: {self.ground_truth.shape}")
        print(f"Number of spectral bands: {self.hsi_data.shape[2]}")
        print(f"Number of classes: {len(np.unique(self.ground_truth))}")
        
        return self.hsi_data, self.ground_truth
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
        --------
        Dict containing dataset information
        """
        if self.hsi_data is None:
            raise ValueError("No data loaded. Call load_indian_pines() first.")
            
        height, width, bands = self.hsi_data.shape
        num_pixels = height * width
        num_classes = len(np.unique(self.ground_truth))
        
        return {
            'height': height,
            'width': width,
            'bands': bands,
            'num_pixels': num_pixels,
            'num_classes': num_classes,
            'data_shape': self.hsi_data.shape,
            'gt_shape': self.ground_truth.shape,
            'class_names': self.class_names
        }
    
    def vectorize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize the HSI data for matrix operations.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Data matrix Y (bands x pixels) and vectorized ground truth
        """
        if self.hsi_data is None:
            raise ValueError("No data loaded. Call load_indian_pines() first.")
            
        height, width, bands = self.hsi_data.shape
        
        # Reshape to (bands, pixels)
        Y = self.hsi_data.reshape(-1, bands).T
        gt_vector = self.ground_truth.reshape(-1)
        
        return Y, gt_vector
    
    def extract_class_spectra(self, class_indices: list) -> Tuple[np.ndarray, list]:
        """
        Extract mean spectra for specified classes.
        
        Parameters:
        -----------
        class_indices : list
            List of class indices to extract (1-based indexing)
            
        Returns:
        --------
        Tuple[np.ndarray, list]
            Endmember matrix S (bands x num_classes) and class names
        """
        if self.hsi_data is None:
            raise ValueError("No data loaded. Call load_indian_pines() first.")
            
        Y, gt_vector = self.vectorize_data()
        bands = Y.shape[0]
        
        endmembers = []
        selected_class_names = []
        
        for class_idx in class_indices:
            # Find pixels belonging to this class
            class_mask = gt_vector == class_idx
            if np.sum(class_mask) == 0:
                print(f"Warning: No pixels found for class {class_idx}")
                continue
                
            # Extract pixels for this class
            class_pixels = Y[:, class_mask]
            
            # Compute mean spectrum
            mean_spectrum = np.mean(class_pixels, axis=1)
            endmembers.append(mean_spectrum)
            selected_class_names.append(self.class_names[class_idx])
        
        S = np.column_stack(endmembers)
        print(f"Extracted {S.shape[1]} endmembers with {S.shape[0]} spectral bands")
        
        return S, selected_class_names
    
    def get_rgb_bands(self) -> Tuple[int, int, int]:
        """
        Get approximate RGB band indices for visualization.
        
        Returns:
        --------
        Tuple of band indices for Red, Green, Blue
        """
        # Indian Pines typical RGB bands (approximate)
        # These are rough estimates for 200-band Indian Pines
        num_bands = self.hsi_data.shape[2]
        
        if num_bands == 200:
            # Standard Indian Pines
            red_band = 29    # ~650 nm
            green_band = 19  # ~550 nm  
            blue_band = 9    # ~450 nm
        else:
            # Generic approximation
            red_band = int(0.7 * num_bands)
            green_band = int(0.4 * num_bands)
            blue_band = int(0.1 * num_bands)
            
        return red_band, green_band, blue_band


def create_synthetic_data(height: int = 50, width: int = 50, bands: int = 100, 
                         num_endmembers: int = 3, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic hyperspectral data for testing algorithms.
    
    Parameters:
    -----------
    height, width : int
        Spatial dimensions
    bands : int
        Number of spectral bands
    num_endmembers : int
        Number of endmembers
    noise_level : float
        Standard deviation of Gaussian noise
        
    Returns:
    --------
    Tuple containing synthetic HSI data, true endmembers, and true abundances
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic endmembers
    S_true = np.random.rand(bands, num_endmembers)
    S_true = S_true / np.linalg.norm(S_true, axis=0)  # Normalize
    
    # Generate synthetic abundances (on simplex)
    num_pixels = height * width
    A_true = np.random.dirichlet(np.ones(num_endmembers), num_pixels).T
    
    # Generate synthetic data
    Y_clean = S_true @ A_true
    noise = noise_level * np.random.randn(bands, num_pixels)
    Y = Y_clean + noise
    
    # Reshape to image format
    hsi_synthetic = Y.T.reshape(height, width, bands)
    
    return hsi_synthetic, S_true, A_true