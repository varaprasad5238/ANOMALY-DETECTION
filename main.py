import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import math
import random

# Step 1: Simulate Univariate Data Stream with noise and seasonal patterns
def generate_data_stream(length=1000):
    data_stream = []
    for t in range(length):
        # Simulate regular and seasonal patterns with noise
        regular_pattern = 50 * math.sin(2 * math.pi * t / 100)  # Regular pattern
        seasonal_pattern = 20 * math.sin(2 * math.pi * t / 365)  # Seasonal pattern
        noise = random.gauss(0, 5)  # Random noise

        # Combine components
        data_point = regular_pattern + seasonal_pattern + noise

        # Inject random anomalies
        if random.random() > 0.95:  # 5% chance of anomaly
            data_point *= random.uniform(2, 5)  # Anomalous spike

        data_stream.append(data_point)
    return data_stream

# Step 2: Anomaly detection using Z-Score method
def calculate_mean(data):
    return sum(data) / len(data)

#Calculate the standard deviation of the data
def calculate_std(data, mean):
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

#Calculate Z-score and detect anomalies
def z_score_anomaly_detection(data, threshold=3):
    mean = calculate_mean(data)
    std_dev = calculate_std(data, mean)
    anomalies = []

    for i, x in enumerate(data):
        z_score = (x - mean) / std_dev
        if abs(z_score) > threshold:
            anomalies.append(i)  # Storing the index of anomalies

    return anomalies

# Step 3: Anomaly detection using IQR (Interquartile Range)
def iqr_anomaly_detection(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]
    return anomalies

# Step 4: Moving Average and Residual-based Anomaly Detection
def moving_average_anomaly_detection(data, window_size=50, threshold=3):
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    residuals = data[window_size - 1:] - moving_avg
    z_scores_residuals = zscore(residuals)
    anomalies = np.where(np.abs(z_scores_residuals) > threshold)[0] + (window_size - 1)
    return anomalies


# Step 5: Function to implement k-NN anomaly detection

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def knn_anomaly_detection(data, k, threshold):
    n = len(data)
    distances = np.zeros((n, n))

    #Calculate the pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(data[i], data[j])
            distances[i][j] = distance
            distances[j][i] = distance  # Symmetric matrix

    #Find the k nearest neighbors for each point
    avg_distances = []
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        k_nearest_neighbors = sorted_distances[1:k + 1]  # Skip the distance to itself (0)
        avg_distance = np.mean(k_nearest_neighbors)
        avg_distances.append(avg_distance)

    #Detect anomalies (those with average distance above threshold)
    anomalies = []
    for i in range(n):
        if avg_distances[i] > threshold:
            anomalies.append(i)

    return anomalies

# Step 5: Visualization of results
def visualize_anomalies(data, anomalies, title):
    plt.plot(data, label="Data Stream")
    plt.scatter(anomalies, [data[each] for each in anomalies], color="red", label="Anomalies")
    plt.title(title)
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Generate data
    data_stream = generate_data_stream()


    # Z-Score Anomaly Detection
    z_anomalies = z_score_anomaly_detection(data_stream)
    print(f"Z-Score detected {len(z_anomalies)} anomalies.")
    visualize_anomalies(data_stream, z_anomalies, "Z-Score Anomaly Detection")

    # IQR Anomaly Detection
    iqr_anomalies = iqr_anomaly_detection(data_stream)
    print(f"IQR detected {len(iqr_anomalies)} anomalies.")
    visualize_anomalies(data_stream, iqr_anomalies, "IQR Anomaly Detection")

    # Moving Average Anomaly Detection
    ma_anomalies = moving_average_anomaly_detection(data_stream)
    print(f"Moving Average detected {len(ma_anomalies)} anomalies.")
    visualize_anomalies(data_stream, ma_anomalies, "Moving Average Anomaly Detection")

    # KNN Anomaly Detection
    kn_anomalies = knn_anomaly_detection(data_stream,5,1.5)
    print(f"KNN detected {len(kn_anomalies)} anomalies.")
    visualize_anomalies(data_stream, kn_anomalies, "KNN Anomaly Detection")