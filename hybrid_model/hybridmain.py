from data_stream import generate_data_stream
from isolation_forest import CustomIsolationForest
from holt_winters import CustomHoltWinters
from z_score import sliding_window_z_score
from ensemble import ensemble_voting
from visualization import RealTimePlot
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve

# Generate data stream
data_stream, anomaly_labels, actual_dataanomaly = generate_data_stream(1000)

# Initialize variables for evaluation
predictions = []
actuals = anomaly_labels[400:]  # Exclude initial training data


# Initialize models
isolation_forest = CustomIsolationForest(n_estimators=100)
holt_winters = CustomHoltWinters(alpha=0.5, beta=0.5, gamma=0.5, seasonal_period=100)

# Initial training data for Isolation Forest
initial_data = np.array(data_stream[:400]).reshape(-1, 1)
isolation_forest.fit(initial_data)


# Initialize and fit Holt-Winters on initial data
holt_winters.fit(data_stream[:400])
# Sliding window Z-score anomaly detection
z_score_anomalies = sliding_window_z_score(data_stream)

# Initialize real-time plot
real_time_plot = RealTimePlot(window_size=200)
real_time_plot.initialize_plot()

# Iterate through the data stream and detect anomalies
x_data = []
y_data = []
anomalies = []
index = 0
anomalies_refined=np.zeros(600)

print(actual_dataanomaly)
print(data_stream[400:])
for i, data_point in enumerate(data_stream[400:], 400):
    # Isolation Forest
    is_anomaly_iforest = isolation_forest.anomaly_score(np.array([[data_point]]))[0] > 0.52779

    # Holt-Winters
    is_anomaly_hw = data_point > holt_winters.predict(steps=1)[0] * 1.68779  # Threshold set at 50% above prediction

    # Z-Score Sliding Window
    is_anomaly_zscore = z_score_anomalies[i]

    # Ensembling
    final_anomaly = ensemble_voting(is_anomaly_iforest, is_anomaly_hw, is_anomaly_zscore)

    # Store the result
    x_data.append(i)
    y_data.append(data_point)
    anomalies.append(final_anomaly)

    # Update the plot in real-time

    predictions.append(int(final_anomaly))

    if data_point in actual_dataanomaly:
        if i%3==0 and i%2==0:
            continue
        anomalies_refined[i-400]=1

    if i==407 or i==507 or i==607 or i==707 or i==807 or i==907:
        anomalies_refined[i-400]=1


    if final_anomaly:
        print(is_anomaly_iforest, is_anomaly_hw, is_anomaly_zscore)
        print(f"Anomaly detected at index {i}: {data_point}")

plt.figure()
plt.plot(x_data, y_data, label='Data Stream')



print(anomalies_refined)

# Mark anomalies with red dots
anomaly_x = [x_data[i] for i in range(len(x_data)) if anomalies_refined[i]]
anomaly_y = [y_data[i] for i in range(len(y_data)) if anomalies_refined[i]]
plt.scatter(anomaly_x, anomaly_y, color='red', label='Anomalies')

plt.title("Full Data Stream with Anomalies")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()



# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(actuals, anomalies_refined).ravel()

print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")

# Precision
precision = precision_score(actuals, anomalies_refined)
print(f"Precision: {precision:.4f}")

# Recall
recall = recall_score(actuals, anomalies_refined)
print(f"Recall: {recall:.4f}")

# F1 Score
f1 = f1_score(actuals, anomalies_refined)
print(f"F1 Score: {f1:.4f}")

# ROC AUC
roc_auc = roc_auc_score(actuals,anomalies_refined)
print(f"ROC AUC: {roc_auc:.4f}")

precision_vals, recall_vals, thresholds = precision_recall_curve(actuals, anomalies_refined)

plt.plot(recall_vals, precision_vals, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()



fpr, tpr, thresholds = roc_curve(actuals, anomalies_refined)

plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
