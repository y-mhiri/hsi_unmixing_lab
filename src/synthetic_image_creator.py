"""
Create synthetic hyperspectral images with basic contiguous shapes and known abundances.
Perfect for testing unmixing algorithms with controlled ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Union
from scipy import ndimage
import matplotlib.patches as patches


class SyntheticImageCreator:
    """Create synthetic hyperspectral images with geometric shapes and known abundances."""
    
    def __init__(self, height: int, width: int):
        """
        Initialize the synthetic image creator.
        
        Parameters:
        -----------
        height : int
            Image height in pixels
        width : int
            Image width in pixels
        """
        self.height = height
        self.width = width
        self.abundance_maps = None
        
        self.shape_info = []
        
    def create_synthetic_image(self, S: np.ndarray, 
                              num_shapes: int = 5,
                              shape_types: Optional[List[str]] = None,
                              noise_level: float = 0.02,
                              background_abundance: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create a synthetic hyperspectral image with basic geometric shapes.
        
        Parameters:
        -----------
        S : np.ndarray
            Endmember spectra matrix (bands, num_endmembers)
        num_shapes : int
            Number of shapes to create
        shape_types : Optional[List[str]]
            Types of shapes to use: 'rectangle', 'circle', 'ellipse'
        noise_level : float
            Standard deviation of Gaussian noise to add
        background_abundance : Optional[np.ndarray]
            Background abundance values (if None, uses uniform background)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, Dict]
            - Synthetic HSI image (height, width, bands)
            - True abundance maps (num_endmembers, height, width)
            - Shape information dictionary
        """
        bands, num_endmembers = S.shape
        
        # Default shape types
        if shape_types is None:
            shape_types = ['rectangle', 'circle', 'ellipse']
        
        # Initialize abundance maps
        self.abundance_maps = np.zeros((num_endmembers, self.height, self.width))
        
        # Create background
        self._create_background(num_endmembers, background_abundance)
        
        # Create shapes with different abundances
        self._create_shapes(num_shapes, shape_types, num_endmembers)
        
        # Generate hyperspectral image using linear mixing model
        hsi_image = self._mix_spectra(S, noise_level)
        
        noise = np.random.normal(0, noise_level, hsi_image.shape)
        hsi_noisy = hsi_image + noise
        # # Prepare shape information
        # shape_info = {
        #     'num_shapes': len(self.shape_info),
        #     'shapes': self.shape_info,
        #     'endmember_names': [f'Endmember_{i+1}' for i in range(num_endmembers)],
        #     'noise_level': noise_level
        # }
        
        return hsi_image, self.abundance_maps, hsi_noisy
    
    def _create_background(self, num_endmembers: int, 
                          background_abundance: Optional[np.ndarray] = None) -> None:
        """Create uniform or specified background abundances."""
        if background_abundance is not None:
            if len(background_abundance) != num_endmembers:
                raise ValueError("Background abundance length must match number of endmembers")
            if not np.isclose(np.sum(background_abundance), 1.0):
                background_abundance = background_abundance / np.sum(background_abundance)
            
            for i in range(num_endmembers):
                self.abundance_maps[i, :, :] = background_abundance[i]
        else:
            # Uniform background (equal abundances)
            uniform_value = 1.0 / num_endmembers
            self.abundance_maps[:, :, :] = uniform_value
    
    def _create_shapes(self, num_shapes: int, shape_types: List[str], 
                      num_endmembers: int) -> None:
        """Create random shapes with different abundance patterns."""
        
        for shape_idx in range(num_shapes):
            # Randomly select shape type
            shape_type = np.random.choice(shape_types)
            
            # Generate random abundance for this shape
            abundance = self._generate_random_abundance(num_endmembers)
            
            # Create shape mask
            mask, shape_params = self._create_shape_mask(shape_type)
            
            # Apply abundance to the shape area
            for i in range(num_endmembers):
                self.abundance_maps[i, mask] = abundance[i]
            
            # Store shape information
            self.shape_info.append({
                'type': shape_type,
                'abundance': abundance,
                'parameters': shape_params,
                'area': np.sum(mask)
            })
    
    def _generate_random_abundance(self, num_endmembers: int, max_index: int = None) -> np.ndarray:
        """Generate random abundance vector on the simplex."""
        # Generate random values
        random_values = np.random.exponential(2.5, num_endmembers)
        
        # Normalize to sum to 1 (project onto simplex)
        abundance = random_values / np.sum(random_values)
        
        return abundance
    
    def _create_shape_mask(self, shape_type: str) -> Tuple[np.ndarray, Dict]:
        """Create a binary mask for the specified shape type."""
        
        if shape_type == 'rectangle':
            return self._create_rectangle_mask()
        elif shape_type == 'circle':
            return self._create_circle_mask()
        elif shape_type == 'ellipse':
            return self._create_ellipse_mask()
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
    
    def _create_rectangle_mask(self) -> Tuple[np.ndarray, Dict]:
        """Create a rectangular shape mask."""
        # Random rectangle parameters
        min_size = min(self.height, self.width) // 8
        max_size = min(self.height, self.width) // 3
        
        width = np.random.randint(min_size, max_size)
        height = np.random.randint(min_size, max_size)
        
        # Random position (ensure shape fits in image)
        x = np.random.randint(0, max(1, self.width - width))
        y = np.random.randint(0, max(1, self.height - height))
        
        # Create mask
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[y:y+height, x:x+width] = True
        
        params = {'x': x, 'y': y, 'width': width, 'height': height}
        
        return mask, params
    
    def _create_circle_mask(self) -> Tuple[np.ndarray, Dict]:
        """Create a circular shape mask."""
        # Random circle parameters
        min_radius = min(self.height, self.width) // 12
        max_radius = min(self.height, self.width) // 6
        
        radius = np.random.randint(min_radius, max_radius)
        
        # Random center (ensure circle fits in image)
        center_x = np.random.randint(radius, self.width - radius)
        center_y = np.random.randint(radius, self.height - radius)
        
        # Create mask
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
        
        params = {'center_x': center_x, 'center_y': center_y, 'radius': radius}
        
        return mask, params
    
    def _create_ellipse_mask(self) -> Tuple[np.ndarray, Dict]:
        """Create an elliptical shape mask."""
        # Random ellipse parameters
        min_axis = min(self.height, self.width) // 12
        max_axis = min(self.height, self.width) // 6
        
        a = np.random.randint(min_axis, max_axis)  # semi-major axis
        b = np.random.randint(min_axis, max_axis)  # semi-minor axis
        
        # Random center (ensure ellipse fits in image)
        center_x = np.random.randint(max(a, b), self.width - max(a, b))
        center_y = np.random.randint(max(a, b), self.height - max(a, b))
        
        # Random rotation angle
        angle = np.random.uniform(0, 2*np.pi)
        
        # Create mask
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        
        # Translate to center
        x_centered = x_grid - center_x
        y_centered = y_grid - center_y
        
        # Rotate coordinates
        x_rot = x_centered * np.cos(angle) + y_centered * np.sin(angle)
        y_rot = -x_centered * np.sin(angle) + y_centered * np.cos(angle)
        
        # Ellipse equation
        mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1
        
        params = {
            'center_x': center_x, 'center_y': center_y,
            'semi_major': a, 'semi_minor': b, 'angle': angle
        }
        
        return mask, params
    
    def _mix_spectra(self, S: np.ndarray, noise_level: float) -> np.ndarray:
        """Mix endmember spectra according to abundance maps."""
        bands, num_endmembers = S.shape
        
        # Initialize hyperspectral image
        hsi_image = np.zeros((self.height, self.width, bands))
        
        # Apply linear mixing model: Y = SA + N
        for i in range(self.height):
            for j in range(self.width):
                # Get abundance vector for this pixel
                abundance_vector = self.abundance_maps[:, i, j]
                
                # Mix spectra
                mixed_spectrum = S @ abundance_vector
                hsi_image[i, j, :] = mixed_spectrum
        
        return hsi_image
    
    def visualize_synthetic_image(self, hsi_image: np.ndarray, 
                                 abundance_maps: np.ndarray,
                                 shape_info: Dict,
                                 rgb_bands: Tuple[int, int, int] = None) -> None:
        """
        Visualize the created synthetic image and abundance maps.
        
        Parameters:
        -----------
        hsi_image : np.ndarray
            Synthetic hyperspectral image
        abundance_maps : np.ndarray
            True abundance maps
        shape_info : Dict
            Information about created shapes
        rgb_bands : Tuple[int, int, int]
            Band indices for RGB visualization
        """
        num_endmembers = abundance_maps.shape[0]
        
        # Determine RGB bands if not specified
        if rgb_bands is None:
            bands = hsi_image.shape[2]
            rgb_bands = (bands//4, bands//2, 3*bands//4)
        
        # Create figure with subplots
        cols = min(4, num_endmembers + 1)
        rows = (num_endmembers + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        # RGB composite
        rgb_image = self._create_rgb_composite(hsi_image, rgb_bands)
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('RGB Composite')
        axes[0, 0].axis('off')
        
        # Abundance maps
        plot_idx = 1
        for i in range(num_endmembers):
            row = plot_idx // cols
            col = plot_idx % cols
            
            im = axes[row, col].imshow(abundance_maps[i], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f'Endmember {i+1} Abundance')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            plot_idx += 1
        
        # Hide extra subplots
        total_plots = num_endmembers + 1
        for idx in range(total_plots, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Synthetic Image with {shape_info["num_shapes"]} Shapes', 
                     fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print shape information
        self._print_shape_info(shape_info)
    
    def _create_rgb_composite(self, hsi_image: np.ndarray, 
                             rgb_bands: Tuple[int, int, int]) -> np.ndarray:
        """Create RGB composite from hyperspectral image."""
        r_band, g_band, b_band = rgb_bands
        
        red = hsi_image[:, :, r_band]
        green = hsi_image[:, :, g_band]
        blue = hsi_image[:, :, b_band]
        
        rgb = np.stack([red, green, blue], axis=2)
        
        # Normalize to [0, 1]
        rgb = rgb - rgb.min()
        rgb = rgb / rgb.max()
        
        return rgb
    
    def _print_shape_info(self, shape_info: Dict) -> None:
        """Print information about created shapes."""
        print(f"\nSynthetic Image Summary:")
        print(f"  Number of shapes: {shape_info['num_shapes']}")
        print(f"  Noise level: {shape_info['noise_level']:.3f}")
        print(f"  Endmembers: {len(shape_info['endmember_names'])}")
        
        print(f"\nShape Details:")
        for i, shape in enumerate(shape_info['shapes']):
            print(f"  Shape {i+1}: {shape['type']}")
            print(f"    Area: {shape['area']} pixels")
            abundance_str = ', '.join([f'{a:.3f}' for a in shape['abundance']])
            print(f"    Abundance: [{abundance_str}]")
            if shape['type'] == 'rectangle':
                p = shape['parameters']
                print(f"    Position: ({p['x']}, {p['y']}), Size: {p['width']}×{p['height']}")
            elif shape['type'] == 'circle':
                p = shape['parameters']
                print(f"    Center: ({p['center_x']}, {p['center_y']}), Radius: {p['radius']}")
            elif shape['type'] == 'ellipse':
                p = shape['parameters']
                print(f"    Center: ({p['center_x']}, {p['center_y']}), Axes: {p['semi_major']}×{p['semi_minor']}")


def create_simple_synthetic_image(S: np.ndarray, 
                                 height: int = 100, 
                                 width: int = 100,
                                 num_shapes: int = 5,
                                 noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simple function to create synthetic hyperspectral image.
    
    Parameters:
    -----------
    S : np.ndarray
        Endmember spectra matrix (bands, num_endmembers)
    height : int
        Image height
    width : int
        Image width
    num_shapes : int
        Number of shapes to create
    noise_level : float
        Noise level to add
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Dict]
        HSI image, abundance maps, shape info
    """
    creator = SyntheticImageCreator(height, width)
    
    hsi_image, abundance_maps, shape_info = creator.create_synthetic_image(
        S, num_shapes=num_shapes, noise_level=noise_level)
    
    # Visualize results
    creator.visualize_synthetic_image(hsi_image, abundance_maps, shape_info)
    
    return hsi_image, abundance_maps, shape_info


def create_custom_synthetic_image(S: np.ndarray,
                                 height: int = 100,
                                 width: int = 100,
                                 shape_configs: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create synthetic image with custom shape configurations.
    
    Parameters:
    -----------
    S : np.ndarray
        Endmember spectra matrix (bands, num_endmembers)
    height : int
        Image height
    width : int
        Image width
    shape_configs : List[Dict]
        Custom shape configurations with 'type', 'position', 'size', 'abundance'
        
    Example:
    --------
    shape_configs = [
        {
            'type': 'rectangle',
            'position': (20, 20),
            'size': (30, 40),
            'abundance': [0.8, 0.1, 0.1]
        },
        {
            'type': 'circle',
            'position': (70, 70),
            'radius': 15,
            'abundance': [0.1, 0.8, 0.1]
        }
    ]
    """
    bands, num_endmembers = S.shape
    
    # Initialize abundance maps with uniform background
    abundance_maps = np.ones((num_endmembers, height, width)) / num_endmembers
    
    shape_info = {'num_shapes': 0, 'shapes': [], 'endmember_names': [f'EM_{i+1}' for i in range(num_endmembers)]}
    
    if shape_configs:
        for config in shape_configs:
            mask = _create_custom_shape_mask(config, height, width)
            abundance = np.array(config['abundance'])
            
            # Normalize abundance
            abundance = abundance / np.sum(abundance)
            
            # Apply to abundance maps
            for i in range(num_endmembers):
                abundance_maps[i, mask] = abundance[i]
            
            shape_info['shapes'].append({
                'type': config['type'],
                'abundance': abundance,
                'area': np.sum(mask),
                'config': config
            })
            shape_info['num_shapes'] += 1
    
    # Mix spectra
    hsi_image = np.zeros((height, width, bands))
    for i in range(height):
        for j in range(width):
            mixed_spectrum = S @ abundance_maps[:, i, j]
            hsi_image[i, j, :] = mixed_spectrum
    
    return hsi_image, abundance_maps, shape_info


def _create_custom_shape_mask(config: Dict, height: int, width: int) -> np.ndarray:
    """Create mask for custom shape configuration."""
    mask = np.zeros((height, width), dtype=bool)
    
    if config['type'] == 'rectangle':
        x, y = config['position']
        w, h = config['size']
        mask[y:y+h, x:x+w] = True
    
    elif config['type'] == 'circle':
        center_x, center_y = config['position']
        radius = config['radius']
        
        y_grid, x_grid = np.ogrid[:height, :width]
        mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
    
    return mask


# Example usage and testing
if __name__ == "__main__":
    # Create example endmember spectra
    bands = 100
    num_endmembers = 3
    
    # Create synthetic endmember spectra with different characteristics
    wavelengths = np.linspace(400, 2500, bands)
    
    S = np.zeros((bands, num_endmembers))
    
    # Endmember 1: High visible, low NIR
    S[:, 0] = 0.3 + 0.4 * np.exp(-(wavelengths - 500)**2 / (2 * 100**2))
    
    # Endmember 2: Low visible, high NIR
    S[:, 1] = 0.1 + 0.6 * (wavelengths > 700) * np.exp(-(wavelengths - 1200)**2 / (2 * 300**2))
    
    # Endmember 3: Moderate across spectrum with absorption
    S[:, 2] = 0.4 - 0.2 * np.exp(-(wavelengths - 1500)**2 / (2 * 200**2))
    
    # Ensure non-negative
    S = np.maximum(S, 0.05)
    
    print("Creating synthetic hyperspectral image...")
    print(f"Endmember spectra shape: {S.shape}")
    
    # Create synthetic image
    hsi_image, abundance_maps, shape_info = create_simple_synthetic_image(
        S, height=120, width=120, num_shapes=6, noise_level=0.03)
    
    print(f"\nSynthetic image created:")
    print(f"  HSI shape: {hsi_image.shape}")
    print(f"  Abundance maps shape: {abundance_maps.shape}")
    print(f"  Number of shapes: {shape_info['num_shapes']}")