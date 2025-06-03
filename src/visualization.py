"""
Visualization utilities for hyperspectral data and unmixing results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple, Optional, List
import seaborn as sns


class HSIVisualizer:
    """
    A class for visualizing hyperspectral data and unmixing results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        
    def plot_band_images(self, hsi_data: np.ndarray, band_indices: List[int], 
                        titles: Optional[List[str]] = None) -> None:
        """
        Plot specific band images from hyperspectral data.
        
        Parameters:
        -----------
        hsi_data : np.ndarray
            HSI data with shape (H, W, B)
        band_indices : List[int]
            List of band indices to display
        titles : Optional[List[str]]
            Optional titles for each band
        """
        num_bands = len(band_indices)
        cols = min(4, num_bands)
        rows = (num_bands + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for i, band_idx in enumerate(band_indices):
            if i < len(axes):
                img = hsi_data[:, :, band_idx]
                im = axes[i].imshow(img, cmap='gray')
                
                title = titles[i] if titles and i < len(titles) else f'Band {band_idx}'
                axes[i].set_title(title)
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(len(band_indices), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_rgb_composite(self, hsi_data: np.ndarray, rgb_bands: Tuple[int, int, int],
                          title: str = "RGB Composite") -> None:
        """
        Create and display RGB composite image.
        
        Parameters:
        -----------
        hsi_data : np.ndarray
            HSI data with shape (H, W, B)
        rgb_bands : Tuple[int, int, int]
            Band indices for Red, Green, Blue
        title : str
            Plot title
        """
        red_idx, green_idx, blue_idx = rgb_bands
        
        # Extract RGB bands
        red = hsi_data[:, :, red_idx]
        green = hsi_data[:, :, green_idx]
        blue = hsi_data[:, :, blue_idx]
        
        # Normalize to [0, 1]
        rgb_image = np.stack([red, green, blue], axis=2)
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def plot_abundance(self, abundance: np.ndarray, position: List[int]=None, wavelengths: np.ndarray = None) -> None:
        """
        Plot abundance as a function of endmember for a specific pixel

        Parameters:
        -----------

        abundance : np.ndarray
            Abundance map
        position : List[int]
            Position of the pixel in the image. If None, center pixel is displayed
        
        """

        nc, h,w = abundance.shape

        if position is None:
            position = [[h//2, w//2]]

        x_label = 'Endmember Index'

        plt.figure(figsize=self.figsize)

        spectra = []
        labels = []
        for x,y in position:
            spectra.append(abundance[:,x,y])
            labels.append(f"pixel ({x},{y})")

        num_spectra = len(spectra)
        colors = plt.cm.tab10(np.linspace(0, 1, num_spectra))
        for i in range(num_spectra):
            plt.plot(spectra[i], 
                    label=labels[i] if i < len(labels) else f'Spectrum {i+1}',
                    color=colors[i], linewidth=2)
        
        plt.xlabel(x_label)
        plt.ylabel('Reflectance')
        plt.title("Abundance spectra")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_ground_truth(self, ground_truth: np.ndarray, class_names: List[str]) -> None:
        """
        Plot ground truth classification map.
        
        Parameters:
        -----------
        ground_truth : np.ndarray
            Ground truth labels with shape (H, W)
        class_names : List[str]
            Names of the classes
        """
        num_classes = len(np.unique(ground_truth))
        
        # Create a colormap
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(ground_truth, cmap=cmap, vmin=0, vmax=num_classes-1)
        plt.title('Ground Truth Classification Map')
        plt.axis('off')
        
        # Create colorbar with class names
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_ticks(range(num_classes))
        cbar.set_ticklabels([class_names[i] if i < len(class_names) else f'Class {i}' 
                            for i in range(num_classes)])
        
        plt.show()
    
    def plot_spectra(self, spectra: np.ndarray, labels: List[str], 
                    wavelengths: Optional[np.ndarray] = None,
                    title: str = "Spectral Signatures") -> None:
        """
        Plot spectral signatures.
        
        Parameters:
        -----------
        spectra : np.ndarray
            Spectral data with shape (bands, num_spectra)
        labels : List[str]
            Labels for each spectrum
        wavelengths : Optional[np.ndarray]
            Wavelength values for x-axis
        title : str
            Plot title
        """
        bands, num_spectra = spectra.shape
        
        if wavelengths is None:
            wavelengths = np.arange(bands)
            x_label = 'Band Index'
        else:
            x_label = 'Wavelength (nm)'
        
        plt.figure(figsize=self.figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, num_spectra))
        
        for i in range(num_spectra):
            plt.plot(wavelengths, spectra[:, i], 
                    label=labels[i] if i < len(labels) else f'Spectrum {i+1}',
                    color=colors[i], linewidth=2)
        
        plt.xlabel(x_label)
        plt.ylabel('Reflectance')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_abundance_maps(self, abundances: np.ndarray, ground_truth: np.ndarray, 
                        shape: Tuple[int, int], endmember_names: List[str], 
                        cmap: Optional[str] = 'hot') -> None:
        """
        Plot abundance maps for each endmember with class boundary overlays.
        
        Parameters:
        -----------
        abundances : np.ndarray
            Abundance matrix with shape (num_endmembers, num_pixels)
        ground_truth : np.ndarray
            Ground truth classification map with shape (height, width)
        shape : Tuple[int, int]
            Spatial dimensions (height, width) to reshape abundances
        endmember_names : List[str]
            Names of endmembers
        cmap : Optional[str]
            Colormap for abundance visualization
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import ndimage
        
        num_endmembers = abundances.shape[0]
        height, width = shape
        
        cols = min(3, num_endmembers)
        rows = (num_endmembers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Ensure ground_truth is the right shape

        if ground_truth is not None:
            if ground_truth.shape != (height, width):
                if ground_truth.size == height * width:
                    ground_truth = ground_truth.reshape(height, width)
                else:
                    raise ValueError(f"Ground truth shape {ground_truth.shape} incompatible with {(height, width)}")
            
        for i in range(num_endmembers):  # Fixed: was num_endmembers-1
            if i < len(axes):
                abundance_map = abundances[i, :].reshape(height, width)
                
                # Plot abundance map
                im = axes[i].imshow(abundance_map, cmap=cmap, vmin=0, vmax=1)
                
                # Add class boundary contours
                # Create contours for each class
                unique_classes = np.unique(ground_truth)
                unique_classes = unique_classes[unique_classes > 0]  # Remove background class 0
                

                # Plot contours for class boundaries
                if ground_truth is not None:
                    class_mask = (ground_truth == i).astype(float)
                        
                        # Smooth the mask slightly to get better contours
                    class_mask_smooth = ndimage.gaussian_filter(class_mask, sigma=0.5)
                        
                        # Draw contour at 0.5 level (class boundary)
                    contour = axes[i].contour(class_mask_smooth, levels=[0.5], 
                                                colors='red', linewidths=1.5, alpha=0.8)
                        
            
                # Set title and formatting
                if endmember_names is not None:
                    name = endmember_names[i] if i < len(endmember_names) else f'Endmember {i+1}'
                axes[i].set_title(f'{name} Abundance')
                axes[i].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(num_endmembers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_reconstruction_comparison(self, original: np.ndarray, reconstructed: np.ndarray,
                                     rgb_bands: Tuple[int, int, int]) -> None:
        """
        Compare original and reconstructed RGB composites.
        
        Parameters:
        -----------
        original : np.ndarray
            Original HSI data with shape (H, W, B)
        reconstructed : np.ndarray
            Reconstructed HSI data with shape (H, W, B)
        rgb_bands : Tuple[int, int, int]
            Band indices for RGB composition
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original RGB
        red_idx, green_idx, blue_idx = rgb_bands
        orig_rgb = np.stack([original[:, :, red_idx], 
                            original[:, :, green_idx], 
                            original[:, :, blue_idx]], axis=2)
        orig_rgb = (orig_rgb - orig_rgb.min()) / (orig_rgb.max() - orig_rgb.min())
        
        # Reconstructed RGB
        recon_rgb = np.stack([reconstructed[:, :, red_idx], 
                             reconstructed[:, :, green_idx], 
                             reconstructed[:, :, blue_idx]], axis=2)
        recon_rgb = (recon_rgb - recon_rgb.min()) / (recon_rgb.max() - recon_rgb.min())
        
        # Difference
        diff_rgb = np.abs(orig_rgb - recon_rgb)
        
        axes[0].imshow(orig_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(recon_rgb)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        im = axes[2].imshow(diff_rgb)
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self, objective_values: List[float], title: str = "Convergence Plot") -> None:
        """
        Plot convergence of optimization algorithm.
        
        Parameters:
        -----------
        objective_values : List[float]
            Objective function values at each iteration
        title : str
            Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.semilogy(objective_values, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value (log scale)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_abundance_statistics(self, abundances: np.ndarray, endmember_names: List[str]) -> None:
        """
        Plot statistics of abundance values.
        
        Parameters:
        -----------
        abundances : np.ndarray
            Abundance matrix with shape (num_endmembers, num_pixels)
        endmember_names : List[str]
            Names of endmembers
        """
        num_endmembers = abundances.shape[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Box plot of abundances
        abundance_data = [abundances[i, :] for i in range(num_endmembers)]
        labels = [endmember_names[i] if i < len(endmember_names) else f'EM{i+1}' 
                 for i in range(num_endmembers)]
        
        axes[0].boxplot(abundance_data, labels=labels)
        axes[0].set_title('Abundance Distribution by Endmember')
        axes[0].set_ylabel('Abundance Value')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Histogram of sum of abundances
        abundance_sums = np.sum(abundances, axis=0)
        axes[1].hist(abundance_sums, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Sum = 1')
        axes[1].set_xlabel('Sum of Abundances')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Abundance Sums')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()