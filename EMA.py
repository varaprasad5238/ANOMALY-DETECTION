import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve


# Step 1: Generate Data Stream and Ground Truth
def generate_data_stream(length=1000):
    data_stream = []
    ground_truth = np.zeros(length)  # To store ground truth anomaly labels (0 = normal, 1 = anomaly)

    for t in range(length):
        regular_pattern = 50 * math.sin(2 * math.pi * t / 100)  # Regular pattern
        seasonal_pattern = 20 * math.sin(2 * math.pi * t / 365)  # Seasonal pattern
        noise = random.gauss(0, 5)  # Random noise
        data_point = regular_pattern + seasonal_pattern + noise

        # Inject random anomalies (5% chance)
        if random.random() > 0.95:
            data_point *= random.uniform(2, 5)
            ground_truth[t] = 1  # Mark this point as an anomaly

        data_stream.append(data_point)

    return np.array(data_stream), ground_truth


# Step 2: Calculate Exponential Moving Average (EMA)
def calculate_ema(data, alpha):
    ema = []
    ema.append(data[0])  # Set the first EMA value as the first data point
    for t in range(1, len(data)):
        ema_value = alpha * data[t] + (1 - alpha) * ema[-1]
        ema.append(ema_value)
    return np.array(ema)


# Step 3: Detect Anomalies Based on EMA
def detect_anomalies_ema(data, ema, threshold=3):
    residuals = np.abs(data - ema)  # Absolute difference between actual data and EMA
    anomalies=[]
    to_return=[]
    for i,x in enumerate(residuals):
        if x>threshold:
            anomalies.append(i)
            to_return.append(1)
        else:
            to_return.append(0)

    return anomalies,to_return


# Step 4: Evaluation Metrics (Precision, Recall, F1-Score)
def evaluate_anomalies(ground_truth, predicted_anomalies, data_length):
    y_true = ground_truth
    y_pred = np.zeros(data_length)
    y_pred[predicted_anomalies] = 1  # Mark predicted anomalies in y_pred

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

def to_plot(actuals,predictions):

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


# Step 5: Visualization
def plot_ema_anomalies(data_stream, ema, anomalies):
    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label="Data Stream")
    plt.plot(ema, label="Exponential Moving Average", color='orange')
    plt.scatter(anomalies, data_stream[anomalies], color='red', label="Anomalies")
    plt.title("Exponential Moving Average Anomaly Detection")
    plt.legend()
    plt.show()


# Main script to run all steps
if __name__ == "__main__":
    # Step 1: Generate Data Stream
    data_stream, ground_truth = generate_data_stream(length=1000)

    # Step 2: Calculate Exponential Moving Average (EMA)
    alpha = 0.1  # Smoothing factor (between 0 and 1)
    ema = calculate_ema(data_stream, alpha)

    # Step 3: Detect Anomalies Based on EMA
    threshold = 40  # Adjust threshold to flag anomalies
    anomalies,to_return = detect_anomalies_ema(data_stream, ema, threshold)

    # Step 4: Evaluate Anomalies
    precision, recall, f1 = evaluate_anomalies(ground_truth, anomalies, len(data_stream))
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # Step 5: Visualize Results
    plot_ema_anomalies(data_stream, ema, anomalies)
    to_plot(ground_truth,to_return)
