import numpy as np
from sklearn.metrics import confusion_matrix

def multi_cluster_accuracy(true_labels, cluster_labels):
    contingency_matrix = confusion_matrix(true_labels, cluster_labels)
    # Assign each cluster to the class with the maximum count
    cluster_to_class = np.argmax(contingency_matrix, axis=0)
    # Calculate the number of correctly assigned instances
    correct = sum(contingency_matrix[cluster_to_class[j], j] for j in range(contingency_matrix.shape[1]))
    total = contingency_matrix.sum()
    accuracy = correct / total
    return accuracy

# Example usage:
# true_labels = np.array([...])
# cluster_labels = np.array([...])
# print(f"Accuracy: {multi_cluster_accuracy(true_labels, cluster_labels) * 100:.2f}%")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from typing import Callable, List, Optional, Tuple


def compute_distortions_with_cdist(
    data: np.ndarray,
    cluster_range: range = range(1, 11),
    clustering_algorithm: Callable = KMeans,
    algorithm_params: Optional[dict] = None,
    distance_metric: str = 'euclidean'
) -> Tuple[List[int], List[float]]:
    """Compute distortions for a range of cluster numbers using cdist.

    Distortion is defined as the mean of the minimum distances from each
    data point to the nearest cluster center.

    Args:
        data (np.ndarray): The input data for clustering. Shape should be (n_samples, n_features).
        cluster_range (range, optional): The range of cluster numbers to evaluate. Defaults to range(1, 11).
        clustering_algorithm (Callable, optional): The clustering algorithm to use. Must have `fit` and `predict` methods.
            Defaults to KMeans.
        algorithm_params (Optional[dict], optional): Additional parameters to pass to the clustering algorithm. Defaults to None.
        distance_metric (str, optional): The distance metric to use for computing distortions. Defaults to 'euclidean'.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing a list of cluster numbers and their corresponding distortions.
    """
    if algorithm_params is None:
        algorithm_params = {}

    distortions = []
    cluster_numbers = list(cluster_range)

    for k in cluster_numbers:
        # Initialize the clustering algorithm with the current number of clusters
        try:
            model = clustering_algorithm(n_clusters=k, **algorithm_params)
        except TypeError as e:
            raise ValueError(
                f"Error initializing {clustering_algorithm.__name__} with params {algorithm_params}: {e}"
            )

        # Fit the model to the data
        model.fit(data)

        # Predict cluster assignments
        cluster_assignments = model.predict(data)

        # Compute the distortion using cdist
        # cdist computes the distance between each data point and each cluster center
        distances = cdist(data, model.cluster_centers_, metric=distance_metric)
        # For each data point, find the minimum distance to any cluster center
        min_distances = np.min(distances, axis=1)
        # Compute the mean distortion
        mean_distortion = np.mean(min_distances)
        distortions.append(mean_distortion)

        print(f"Computed distortion for k={k}: {mean_distortion:.4f}")

    return cluster_numbers, distortions


def plot_elbow_with_cdist(
    cluster_numbers: List[int],
    distortions: List[float],
    title: str = 'Elbow Method (Using cdist)',
    xlabel: str = 'Number of Clusters (k)',
    ylabel: str = 'Mean Distortion',
    show: bool = True,
    save_path: Optional[str] = None
) -> None:
    """Plot the elbow curve for evaluating the optimal number of clusters.

    Args:
        cluster_numbers (List[int]): The list of cluster numbers evaluated.
        distortions (List[float]): The corresponding list of distortions for each cluster number.
        title (str, optional): The title of the plot. Defaults to 'Elbow Method (Using cdist)'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Number of Clusters (k)'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Mean Distortion'.
        show (bool, optional): Whether to display the plot. Defaults to True.
        save_path (Optional[str], optional): If provided, the plot will be saved to this path. Defaults to None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_numbers, distortions, 'bo-', markersize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(cluster_numbers)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Elbow plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def find_optimal_clusters_elbow_with_cdist(
    data: np.ndarray,
    cluster_range: range = range(1, 11),
    clustering_algorithm: Callable = KMeans,
    algorithm_params: Optional[dict] = None,
    distance_metric: str = 'euclidean',
    plot: bool = True,
    plot_params: Optional[dict] = None
) -> Tuple[List[int], List[float]]:
    """Find the optimal number of clusters using the Elbow Method with cdist.

    Args:
        data (np.ndarray): The input data for clustering.
        cluster_range (range, optional): The range of cluster numbers to evaluate. Defaults to range(1, 11).
        clustering_algorithm (Callable, optional): The clustering algorithm to use.
            Defaults to KMeans.
        algorithm_params (Optional[dict], optional): Additional parameters for the clustering algorithm.
            Defaults to None.
        distance_metric (str, optional): The distance metric for computing distortions. Defaults to 'euclidean'.
        plot (bool, optional): Whether to plot the elbow curve. Defaults to True.
        plot_params (Optional[dict], optional): Additional parameters for plotting. Defaults to None.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing cluster numbers and their corresponding distortions.
    """
    if plot_params is None:
        plot_params = {}

    cluster_numbers, distortions = compute_distortions_with_cdist(
        data=data,
        cluster_range=cluster_range,
        clustering_algorithm=clustering_algorithm,
        algorithm_params=algorithm_params,
        distance_metric=distance_metric
    )

    if plot:
        plot_elbow_with_cdist(cluster_numbers, distortions, **plot_params)

    return cluster_numbers, distortions


def main_teaching_example():
    """Run the teaching example using cdist to compute distortions."""
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from sklearn.datasets import make_blobs

    # Generate synthetic data for demonstration
    data, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Define clustering parameters
    clustering_params = {
        'n_init': 'auto',
        'random_state': 42
    }

    # Find optimal clusters using the elbow method with cdist
    cluster_nums, distortions = find_optimal_clusters_elbow_with_cdist(
        data=data_scaled,
        cluster_range=range(1, 10),
        clustering_algorithm=KMeans,
        algorithm_params=clustering_params,
        distance_metric='euclidean',
        plot=True,
        plot_params={
            'title': 'Elbow Method for Optimal k (Teaching Example)',
            'xlabel': 'Number of Clusters (k)',
            'ylabel': 'Mean Distortion',
            'show': True,
            'save_path': None  # e.g., 'elbow_plot_teaching.png'
        }
    )

    # Choose the optimal number of clusters (for example, 3)
    optimal_k = 3
    final_model = KMeans(n_clusters=optimal_k, **clustering_params)
    final_model.fit(data_scaled)
    final_predictions = final_model.predict(data_scaled)

    # Add predictions to a DataFrame for further analysis
    df = pd.DataFrame(data_scaled, columns=['Feature1', 'Feature2'])
    df['Cluster'] = final_predictions
    print("\nSample of Cluster Assignments:")
    print(df.head())


if __name__ == "__main__":
    main_teaching_example()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from typing import Callable, List, Optional, Tuple


def compute_distortions_with_cdist(
    data: np.ndarray,
    cluster_range: range = range(1, 11),
    clustering_algorithm: Callable = KMeans,
    algorithm_params: Optional[dict] = None,
    distance_metric: str = 'euclidean'
) -> Tuple[List[int], List[float]]:
    """Compute distortions for a range of cluster numbers using cdist.

    Distortion is defined as the mean of the minimum distances from each
    data point to the nearest cluster center.

    Args:
        data (np.ndarray): The input data for clustering. Shape should be (n_samples, n_features).
        cluster_range (range, optional): The range of cluster numbers to evaluate. Defaults to range(1, 11).
        clustering_algorithm (Callable, optional): The clustering algorithm to use. Must have `fit` and `predict` methods.
            Defaults to KMeans.
        algorithm_params (Optional[dict], optional): Additional parameters to pass to the clustering algorithm. Defaults to None.
        distance_metric (str, optional): The distance metric to use for computing distortions. Defaults to 'euclidean'.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing a list of cluster numbers and their corresponding distortions.
    """
    if algorithm_params is None:
        algorithm_params = {}

    distortions = []
    cluster_numbers = list(cluster_range)

    for k in cluster_numbers:
        # Initialize the clustering algorithm with the current number of clusters
        try:
            model = clustering_algorithm(n_clusters=k, **algorithm_params)
        except TypeError as e:
            raise ValueError(
                f"Error initializing {clustering_algorithm.__name__} with params {algorithm_params}: {e}"
            )

        # Fit the model to the data
        model.fit(data)

        # Predict cluster assignments
        cluster_assignments = model.predict(data)

        # Compute the distortion using cdist
        # cdist computes the distance between each data point and each cluster center
        distances = cdist(data, model.cluster_centers_, metric=distance_metric)
        # For each data point, find the minimum distance to any cluster center
        min_distances = np.min(distances, axis=1)
        # Compute the mean distortion
        mean_distortion = np.mean(min_distances)
        distortions.append(mean_distortion)

        print(f"Computed distortion for k={k}: {mean_distortion:.4f}")

    return cluster_numbers, distortions


def plot_elbow_with_cdist(
    cluster_numbers: List[int],
    distortions: List[float],
    title: str = 'Elbow Method (Using cdist)',
    xlabel: str = 'Number of Clusters (k)',
    ylabel: str = 'Mean Distortion',
    show: bool = True,
    save_path: Optional[str] = None,
    optimal_k: Optional[int] = None
) -> None:
    """Plot the elbow curve for evaluating the optimal number of clusters.

    Args:
        cluster_numbers (List[int]): The list of cluster numbers evaluated.
        distortions (List[float]): The corresponding list of distortions for each cluster number.
        title (str, optional): The title of the plot. Defaults to 'Elbow Method (Using cdist)'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Number of Clusters (k)'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Mean Distortion'.
        show (bool, optional): Whether to display the plot. Defaults to True.
        save_path (Optional[str], optional): If provided, the plot will be saved to this path. Defaults to None.
        optimal_k (Optional[int], optional): The optimal number of clusters to highlight on the plot. Defaults to None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_numbers, distortions, 'bo-', markersize=8, label='Distortion')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(cluster_numbers)
    plt.grid(True)

    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Elbow plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def find_elbow_max_distance(
    cluster_numbers: List[int],
    distortions: List[float]
) -> int:
    """Find the elbow point using the Maximum Distance Method.

    This method fits a line between the first and last points and finds the point
    with the maximum perpendicular distance from this line.

    Args:
        cluster_numbers (List[int]): The list of cluster numbers evaluated.
        distortions (List[float]): The corresponding list of distortions for each cluster number.

    Returns:
        int: The optimal number of clusters (elbow point).
    """
    # Coordinates of the first and last points
    x1, y1 = cluster_numbers[0], distortions[0]
    x2, y2 = cluster_numbers[-1], distortions[-1]

    # Compute the line coefficients (Ax + By + C = 0)
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Compute distances from each point to the line
    distances = []
    for x, y in zip(cluster_numbers, distortions):
        distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
        distances.append(distance)

    # Elbow is the point with maximum distance
    optimal_index = np.argmax(distances)
    optimal_k = cluster_numbers[optimal_index]

    print(f"Optimal number of clusters (elbow point) found: k={optimal_k}")
    return optimal_k


def find_optimal_clusters_elbow_with_cdist(
    data: np.ndarray,
    cluster_range: range = range(1, 11),
    clustering_algorithm: Callable = KMeans,
    algorithm_params: Optional[dict] = None,
    distance_metric: str = 'euclidean',
    plot: bool = True,
    plot_params: Optional[dict] = None,
    auto_detect_elbow: bool = True
) -> Tuple[List[int], List[float], Optional[int]]:
    """Find the optimal number of clusters using the Elbow Method with cdist.

    Args:
        data (np.ndarray): The input data for clustering.
        cluster_range (range, optional): The range of cluster numbers to evaluate. Defaults to range(1, 11).
        clustering_algorithm (Callable, optional): The clustering algorithm to use.
            Defaults to KMeans.
        algorithm_params (Optional[dict], optional): Additional parameters for the clustering algorithm.
            Defaults to None.
        distance_metric (str, optional): The distance metric for computing distortions. Defaults to 'euclidean'.
        plot (bool, optional): Whether to plot the elbow curve. Defaults to True.
        plot_params (Optional[dict], optional): Additional parameters for plotting. Defaults to None.
        auto_detect_elbow (bool, optional): Whether to automatically detect the elbow point. Defaults to True.

    Returns:
        Tuple[List[int], List[float], Optional[int]]: A tuple containing cluster numbers, their corresponding distortions,
            and the optimal number of clusters (elbow point) if detected.
    """
    if plot_params is None:
        plot_params = {}

    cluster_numbers, distortions = compute_distortions_with_cdist(
        data=data,
        cluster_range=cluster_range,
        clustering_algorithm=clustering_algorithm,
        algorithm_params=algorithm_params,
        distance_metric=distance_metric
    )

    optimal_k = None
    if auto_detect_elbow:
        optimal_k = find_elbow_max_distance(cluster_numbers, distortions)

    if plot:
        plot_elbow_with_cdist(
            cluster_numbers,
            distortions,
            **plot_params,
            optimal_k=optimal_k
        )

    return cluster_numbers, distortions, optimal_k


def main_teaching_example_with_elbow_detection():
    """Run the teaching example using cdist to compute distortions and detect elbow point."""
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from sklearn.datasets import make_blobs

    # Generate synthetic data for demonstration
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Define clustering parameters
    clustering_params = {
        'n_init': 'auto',
        'random_state': 42
    }

    # Find optimal clusters using the elbow method with cdist
    cluster_nums, distortions, optimal_k = find_optimal_clusters_elbow_with_cdist(
        data=data_scaled,
        cluster_range=range(1, 10),
        clustering_algorithm=KMeans,
        algorithm_params=clustering_params,
        distance_metric='euclidean',
        plot=True,
        plot_params={
            'title': 'Elbow Method for Optimal k (Teaching Example)',
            'xlabel': 'Number of Clusters (k)',
            'ylabel': 'Mean Distortion',
            'show': True,
            'save_path': None  # e.g., 'elbow_plot_teaching.png'
        },
        auto_detect_elbow=True
    )

    # Choose the optimal number of clusters based on elbow detection
    if optimal_k is None:
        optimal_k = 3  # Fallback to a default value if elbow not detected

    final_model = KMeans(n_clusters=optimal_k, **clustering_params)
    final_model.fit(data_scaled)
    final_predictions = final_model.predict(data_scaled)

    # Add predictions to a DataFrame for further analysis
    df = pd.DataFrame(data_scaled, columns=['Feature1', 'Feature2'])
    df['Cluster'] = final_predictions
    print("\nSample of Cluster Assignments:")
    print(df.head())


if __name__ == "__main__":
    main_teaching_example_with_elbow_detection()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Callable, List, Optional, Tuple


def compute_inertia(
    data: np.ndarray,
    cluster_range: range = range(1, 11),
    clustering_algorithm: Callable = KMeans,
    algorithm_params: Optional[dict] = None
) -> Tuple[List[int], List[float]]:
    """Compute inertia for a range of cluster numbers using KMeans' inertia_.

    Inertia is defined as the sum of squared distances of samples to their nearest cluster center.

    Args:
        data (np.ndarray): The input data for clustering. Shape should be (n_samples, n_features).
        cluster_range (range, optional): The range of cluster numbers to evaluate. Defaults to range(1, 11).
        clustering_algorithm (Callable, optional): The clustering algorithm to use. Must have `fit` method.
            Defaults to KMeans.
        algorithm_params (Optional[dict], optional): Additional parameters to pass to the clustering algorithm.
            Defaults to None.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing a list of cluster numbers and their corresponding inertia values.
    """
    if algorithm_params is None:
        algorithm_params = {}

    inertias = []
    cluster_numbers = list(cluster_range)

    for k in cluster_numbers:
        # Initialize the clustering algorithm with the current number of clusters
        try:
            model = clustering_algorithm(n_clusters=k, **algorithm_params)
        except TypeError as e:
            raise ValueError(
                f"Error initializing {clustering_algorithm.__name__} with params {algorithm_params}: {e}"
            )

        # Fit the model to the data
        model.fit(data)

        # Retrieve inertia
        inertia = model.inertia_
        inertias.append(inertia)

        print(f"Computed inertia for k={k}: {inertia:.4f}")

    return cluster_numbers, inertias


def plot_elbow_optimized(
    cluster_numbers: List[int],
    inertias: List[float],
    title: str = 'Elbow Method (Optimized)',
    xlabel: str = 'Number of Clusters (k)',
    ylabel: str = 'Inertia',
    show: bool = True,
    save_path: Optional[str] = None,
    optimal_k: Optional[int] = None
) -> None:
    """Plot the elbow curve for evaluating the optimal number of clusters using inertia.

    Args:
        cluster_numbers (List[int]): The list of cluster numbers evaluated.
        inertias (List[float]): The corresponding list of inertia values for each cluster number.
        title (str, optional): The title of the plot. Defaults to 'Elbow Method (Optimized)'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Number of Clusters (k)'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Inertia'.
        show (bool, optional): Whether to display the plot. Defaults to True.
        save_path (Optional[str], optional): If provided, the plot will be saved to this path. Defaults to None.
        optimal_k (Optional[int], optional): The optimal number of clusters to highlight on the plot. Defaults to None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_numbers, inertias, 'rs-', markersize=8, label='Inertia')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(cluster_numbers)
    plt.grid(True)

    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='b', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Elbow plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def find_elbow_max_distance(
    cluster_numbers: List[int],
    inertias: List[float]
) -> int:
    """Find the elbow point using the Maximum Distance Method.

    This method fits a line between the first and last points and finds the point
    with the maximum perpendicular distance from this line.

    Args:
        cluster_numbers (List[int]): The list of cluster numbers evaluated.
        inertias (List[float]): The corresponding list of inertia values for each cluster number.

    Returns:
        int: The optimal number of clusters (elbow point).
    """
    # Coordinates of the first and last points
    x1, y1 = cluster_numbers[0], inertias[0]
    x2, y2 = cluster_numbers[-1], inertias[-1]

    # Compute the line coefficients (Ax + By + C = 0)
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Compute distances from each point to the line
    distances = []
    for x, y in zip(cluster_numbers, inertias):
        distance = abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
        distances.append(distance)

    # Elbow is the point with maximum distance
    optimal_index = np.argmax(distances)
    optimal_k = cluster_numbers[optimal_index]

    print(f"Optimal number of clusters (elbow point) found: k={optimal_k}")
    return optimal_k


def find_optimal_clusters_elbow_optimized(
    data: np.ndarray,
    cluster_range: range = range(1, 11),
    clustering_algorithm: Callable = KMeans,
    algorithm_params: Optional[dict] = None,
    plot: bool = True,
    plot_params: Optional[dict] = None,
    auto_detect_elbow: bool = True
) -> Tuple[List[int], List[float], Optional[int]]:
    """Find the optimal number of clusters using the Elbow Method with inertia.

    Args:
        data (np.ndarray): The input data for clustering.
        cluster_range (range, optional): The range of cluster numbers to evaluate. Defaults to range(1, 11).
        clustering_algorithm (Callable, optional): The clustering algorithm to use.
            Defaults to KMeans.
        algorithm_params (Optional[dict], optional): Additional parameters for the clustering algorithm.
            Defaults to None.
        plot (bool, optional): Whether to plot the elbow curve. Defaults to True.
        plot_params (Optional[dict], optional): Additional parameters for plotting. Defaults to None.
        auto_detect_elbow (bool, optional): Whether to automatically detect the elbow point. Defaults to True.

    Returns:
        Tuple[List[int], List[float], Optional[int]]: A tuple containing cluster numbers, their corresponding inertia values,
            and the optimal number of clusters (elbow point) if detected.
    """
    if plot_params is None:
        plot_params = {}

    cluster_numbers, inertias = compute_inertia(
        data=data,
        cluster_range=cluster_range,
        clustering_algorithm=clustering_algorithm,
        algorithm_params=algorithm_params
    )

    optimal_k = None
    if auto_detect_elbow:
        optimal_k = find_elbow_max_distance(cluster_numbers, inertias)

    if plot:
        plot_elbow_optimized(
            cluster_numbers,
            inertias,
            **plot_params,
            optimal_k=optimal_k
        )

    return cluster_numbers, inertias, optimal_k


def main_optimized_example_with_elbow_detection():
    """Run the optimized example using KMeans' inertia to compute distortions and detect elbow point."""
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from sklearn.datasets import make_blobs

    # Generate synthetic data for demonstration
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Define clustering parameters
    clustering_params = {
        'n_init': 'auto',
        'random_state': 42
    }

    # Find optimal clusters using the elbow method with inertia
    cluster_nums, inertias, optimal_k = find_optimal_clusters_elbow_optimized(
        data=data_scaled,
        cluster_range=range(1, 10),
        clustering_algorithm=KMeans,
        algorithm_params=clustering_params,
        plot=True,
        plot_params={
            'title': 'Elbow Method for Optimal k (Optimized)',
            'xlabel': 'Number of Clusters (k)',
            'ylabel': 'Inertia',
            'show': True,
            'save_path': None  # e.g., 'elbow_plot_optimized.png'
        },
        auto_detect_elbow=True
    )

    # Choose the optimal number of clusters based on elbow detection
    if optimal_k is None:
        optimal_k = 3  # Fallback to a default value if elbow not detected

    final_model = KMeans(n_clusters=optimal_k, **clustering_params)
    final_model.fit(data_scaled)
    final_predictions = final_model.predict(data_scaled)

    # Add predictions to a DataFrame for further analysis
    df = pd.DataFrame(data_scaled, columns=['Feature1', 'Feature2'])
    df['Cluster'] = final_predictions
    print("\nSample of Cluster Assignments:")
    print(df.head())


if __name__ == "__main__":
    main_optimized_example_with_elbow_detection()

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

# Example true labels and cluster assignments
true_labels = np.array([...])  # Replace with your true class labels (e.g., 0, 1, 2)
cluster_labels = np.array([...])  # Replace with your cluster assignments (e.g., 0, 1, 2)

# Step 1: Create the contingency matrix
contingency_matrix = confusion_matrix(true_labels, cluster_labels)

# Step 2: Apply the Hungarian Algorithm
# Since we want to maximize the matches, we convert it to a cost matrix by subtracting from the max value
cost_matrix = contingency_matrix.max() - contingency_matrix
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Step 3: Calculate the accuracy
total_correct = contingency_matrix[row_ind, col_ind].sum()
total_instances = contingency_matrix.sum()
accuracy = total_correct / total_instances

print(f"Optimal Accuracy: {accuracy * 100:.2f}%")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
np.random.seed(42)

# Define cluster specifications
clusters = {
    0: {
        'mean': [-0.02199, 0.22, 0.22],
        'count': 678,
        'std': [0.005, 0.05, 0.05]  # Assumed standard deviations
    },
    1: {
        'outlier': [-0.18474, -30, 1.049],
        'count': 1
    },
    2: {
        'mean': [-0.00412, 0.061, 0.054],
        'count': 1294,
        'std': [0.005, 0.05, 0.05]
    },
    3: {
        'mean': [-0.01, -8.78e-07, 5.65e-06],
        'count': 855,
        'std': [0.005, 0.05, 0.05]
    },
    4: {
        'mean': [-0.43, -0.186, 0.43],
        'count': 129,
        'std': [0.01, 0.05, 0.05]
    }
}

# Initialize empty lists to store data
data = {
    'slope_iar_dist_to_1st': [],
    'slope_top_3_iar_rank': [],
    'slope_lag_drr': [],
    'cluster': []
}

# Generate data for each cluster
for cluster_id, specs in clusters.items():
    if cluster_id == 1:
        # Add the outlier
        outlier = specs['outlier']
        data['slope_iar_dist_to_1st'].append(outlier[0])
        data['slope_top_3_iar_rank'].append(outlier[1])
        data['slope_lag_drr'].append(outlier[2])
        data['cluster'].append(cluster_id)
    else:
        mean = specs['mean']
        count = specs['count']
        std = specs['std']
        # Generate normally distributed data
        cluster_data = np.random.normal(loc=mean, scale=std, size=(count, 3))
        data['slope_iar_dist_to_1st'].extend(cluster_data[:, 0])
        data['slope_top_3_iar_rank'].extend(cluster_data[:, 1])
        data['slope_lag_drr'].extend(cluster_data[:, 2])
        data['cluster'].extend([cluster_id]*count)

# Create a DataFrame
df = pd.DataFrame(data)

# Shuffle the DataFrame to mix clusters
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display basic information about the generated data
print("Generated DataFrame:")
print(df.head())
print("\nCluster Counts:")
print(df['cluster'].value_counts().sort_index())

