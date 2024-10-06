import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import math
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve

actual_anomalies=[]
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
            actual_anomalies.append(1)
        else:
            actual_anomalies.append(0)

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
    to_return=[]
    for i, x in enumerate(data):
        z_score = (x - mean) / std_dev
        if abs(z_score) > threshold:
            anomalies.append(1)  # Storing the index of anomalies
            to_return.append(i)
        else:
            anomalies.append(0)

    return to_return,anomalies

# Step 3: Anomaly detection using IQR (Interquartile Range)
def iqr_anomaly_detection(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies=[]
    to_return=[]
    for i, x in enumerate(data):
        if (x < lower_bound) | (x > upper_bound):
            anomalies.append(1)  # Storing the index of anomalies
            to_return.append(i)
        else:
            anomalies.append(0)
    return to_return,anomalies,

# Step 4: Moving Average and Residual-based Anomaly Detection
def moving_average_anomaly_detection(data_stream, window_size=50, threshold=3):
    moving_avg = []  # Store the moving average values
    anomalies = []  # List to store indices of detected anomalies
    to_return=[]
    for i in range(len(data_stream)):
        if i < window_size:
            # If not enough data points to fill the window, continue
            moving_avg.append(np.mean(data_stream[:i + 1]))
            to_return.append(0)
        else:
            # Calculate moving average for the current window
            current_window = data_stream[i - window_size + 1:i + 1]
            current_avg = np.mean(current_window)
            moving_avg.append(current_avg)

            # Calculate residual (difference between actual and moving average)
            residual = data_stream[i] - current_avg

            # Calculate the z-score of residuals up to the current point
            residuals = np.array(data_stream[window_size - 1:i + 1]) - np.array(moving_avg[window_size - 1:])
            z_scores_residuals = zscore(residuals)

            # Check if the current z-score exceeds the anomaly threshold
            if np.abs(z_scores_residuals[-1]) > threshold:
                anomalies.append(i)  # Mark as anomaly
                to_return.append(1)
            else:
                to_return.append(0)

    return anomalies,to_return


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
    to_return=[]
    for i in range(n):
        if avg_distances[i] > threshold:
            anomalies.append(i)
            to_return.append(1)
        else:
            to_return.append(0)

    return anomalies,to_return

# Step 5: Visualization of results
def visualize_anomalies(data, anomalies, title):
    plt.plot(data, label="Data Stream")
    plt.scatter(anomalies, [data[each] for each in anomalies], color="red", label="Anomalies")
    plt.title(title)
    plt.legend()
    plt.show()


def to_plot(predictions):
    actuals = actual_anomalies
    # Precision
    precision = precision_score(actuals, predictions)
    print(f"Precision: {precision:.4f}")

    # Recall
    recall = recall_score(actuals, predictions)
    print(f"Recall: {recall:.4f}")

    # F1 Score
    f1 = f1_score(actuals, predictions)
    print(f"F1 Score: {f1:.4f}")

    # ROC AUC
    roc_auc = roc_auc_score(actuals, predictions)
    print(f"ROC AUC: {roc_auc:.4f}")

    precision_vals, recall_vals, thresholds = precision_recall_curve(actuals, predictions)

    plt.plot(recall_vals, precision_vals, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    fpr, tpr, thresholds = roc_curve(actuals, predictions)

    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


# Main function
if __name__ == "__main__":
    # Generate data
    data_stream = generate_data_stream()

    # Z-Score Anomaly Detection
    z_anomalies,anomalies = z_score_anomaly_detection(data_stream)
    print(f"Z-Score detected {len(z_anomalies)} anomalies.")
    visualize_anomalies(data_stream,z_anomalies, "Z-Score Anomaly Detection")
    to_plot(anomalies)

    # IQR Anomaly Detection
    iqr_anomalies,anomalies = iqr_anomaly_detection(data_stream)
    print(f"IQR detected {len(iqr_anomalies)} anomalies.")
    visualize_anomalies(data_stream, iqr_anomalies, "IQR Anomaly Detection")
    to_plot(anomalies)

    # Moving Average Anomaly Detection
    ma_anomalies,anomalies  = moving_average_anomaly_detection(data_stream)
    print(f"Moving Average detected {len(ma_anomalies)} anomalies.")
    visualize_anomalies(data_stream, ma_anomalies, "Moving Average Anomaly Detection")
    to_plot(anomalies)

    # KNN Anomaly Detection
    kn_anomalies,anomalies  = knn_anomaly_detection(data_stream,5,1.5)
    print(f"KNN detected {len(kn_anomalies)} anomalies.")
    visualize_anomalies(data_stream, kn_anomalies, "KNN Anomaly Detection")
    to_plot(anomalies)