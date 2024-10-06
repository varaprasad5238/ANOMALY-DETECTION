import numpy as np


def sliding_window_z_score(data_stream, window_size=50, threshold=2.6879):
    """
    Perform Z-score anomaly detection using a sliding window approach.
    :parameter data_stream: The entire stream of data.
    :parameter window_size: The size of the sliding window.
    :parameter threshold: The Z-score threshold to consider a point an anomaly.
    :return: List of boolean values indicating if the data point is an anomaly.
    """
    anomalies = []
    window = []

    for i, data_point in enumerate(data_stream):
        # Keep a window of the last 'window_size' points
        if len(window) >= window_size:
            window.pop(0)  # Remove the oldest element

        window.append(data_point)

        # Only perform Z-score calculation when we have enough data in the window
        if len(window) < window_size:
            anomalies.append(False)  # Not enough data to make a decision
            continue

        # Calculate mean and standard deviation for the current window
        mean = np.mean(window)
        std = np.std(window)

        # Avoid division by zero
        if std == 0:
            std = 1e-6  # Small constant to avoid division by zero

        # Compute Z-score
        z_score = abs((data_point - mean) / std)

        # Flag as anomaly if Z-score exceeds the threshold
        anomalies.append(z_score > threshold)

    return anomalies

