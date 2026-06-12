import numpy as np

def extract_statistical_features(landmarks):
    """
    Extracts statistical features from raw landmark coordinates for two hands.
    
    Parameters:
    landmarks (list): A flat list of 126 values [x0, y0, z0, ... x41, y41, z41]
                      representing 21 landmarks for Hand 1 and 21 landmarks for Hand 2.
                      
    Returns:
    list: 7 statistical feature values matching your CSV header requirements.
    """
    # Convert to numpy array and reshape to (42 landmarks, 3 coordinates)
    lm_array = np.array(landmarks).reshape(-1, 3)
    
    x_coords = lm_array[:, 0]
    y_coords = lm_array[:, 1]
    z_coords = lm_array[:, 2]
    
    # 1. Aspect Ratio of the combined hand bounding box
    x_max, x_min = np.max(x_coords), np.min(x_coords)
    y_max, y_min = np.max(y_coords), np.min(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Prevent division by zero if coordinates are perfectly flat
    stat_aspect_ratio = width / height if height != 0 else 0.0
    
    # 2. Finger Distance Metrics (Mean, Std, Var)
    # Calculates Euclidean distances of all tracked landmarks relative to the center of mass
    center_of_mass = np.mean(lm_array, axis=0)
    distances = np.linalg.norm(lm_array - center_of_mass, axis=1)
    
    stat_finger_mean = float(np.mean(distances))
    stat_finger_std = float(np.std(distances))
    stat_finger_var = float(np.var(distances))
    
    # 3. Coordinate-wise Spread (Standard Deviation for X, Y, Z)
    stat_x_std = float(np.std(x_coords))
    stat_y_std = float(np.std(y_coords))
    stat_z_std = float(np.std(z_coords))
    
    return [
        stat_aspect_ratio,
        stat_finger_mean,
        stat_finger_std,
        stat_finger_var,
        stat_x_std,
        stat_y_std,
        stat_z_std
    ]