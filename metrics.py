import numpy as np
from sklearn.decomposition import PCA
import torch  # Added for type checking

# Define constants
PCA_DIMENSIONS = 4


def compute_diversity(embeddings):
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        # Ensure it's a NumPy array if it's not a tensor (e.g., already a NumPy array)
        embeddings_np = np.asarray(embeddings)

    n = embeddings_np.shape[0]

    if n < 2:
        # Not enough embeddings to compute diversity metrics meaningfully
        return {
            "semantic_diversity": 0.0,
            "mst_diversity": 0.0,
            "hull_area": 0.0,
        }

    sim_matrix = embeddings_np @ embeddings_np.T

    # Remove self-similarity from diagonal
    sim_matrix[np.arange(n), np.arange(n)] = 0.0  # Use float for consistency

    # Handle n * (n - 1) == 0 case, though n < 2 check above should cover n=0, n=1
    denominator = n * (n - 1)
    if denominator == 0:  # Should not happen if n >= 2
        avg_sim = 0.0
    else:
        avg_sim = sim_matrix.sum() / denominator

    semantic_diversity_val = 1.0 - float(avg_sim)

    # Also calculate minimum spanning tree cost as diversity metric
    from scipy.sparse.csgraph import minimum_spanning_tree

    # Convert similarity to distance
    distance_matrix = 1.0 - sim_matrix
    # Ensure it's symmetric and non-negative
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    np.clip(
        distance_matrix, 0, None, out=distance_matrix
    )  # Ensure distances are non-negative

    # Calculate MST cost
    mst = minimum_spanning_tree(distance_matrix)
    mst_cost_val = float(mst.sum())

    # Calculate hull area if possible
    hull_area_val = 0.0
    coords_for_hull = None
    if n > PCA_DIMENSIONS:  # Need n_samples > n_components for PCA
        try:
            from scipy.spatial import ConvexHull

            # Project to PCA_DIMENSIONS for hull calculation
            pca = PCA(n_components=PCA_DIMENSIONS)
            coords_for_hull = pca.fit_transform(embeddings_np)

            # ConvexHull requires at least d+1 points in d dimensions
            if len(coords_for_hull) >= PCA_DIMENSIONS + 1:
                hull = ConvexHull(coords_for_hull)
                hull_area_val = float(hull.volume)  # In N-D, .volume is the N-D volume
            else:
                hull_area_val = 0.0  # Not enough points for hull after PCA
        except ImportError:  # scipy.spatial might not be available
            hull_area_val = 0.0  # Or handle error as appropriate
            print(
                "Warning: scipy.spatial.ConvexHull could not be imported. Hull area will be 0."
            )
        except Exception as e:
            hull_area_val = 0.0  # Other errors during hull calculation
            print(f"Warning: Could not compute hull area: {e}")
    else:
        # Not enough samples for PCA or meaningful hull
        hull_area_val = 0.0

    # Return metrics dict with hidden coordinates
    metrics = {
        "semantic_diversity": semantic_diversity_val,  # Higher is more diverse
        "mst_diversity": mst_cost_val,  # Higher is more diverse
        "hull_area": hull_area_val,  # Higher is more diverse
    }

    return metrics
