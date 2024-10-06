import math
import random


def generate_data_stream(length=1000):
    """
    Generates a data stream with regular patterns, seasonal variations, noise, and occasional anomalies.
    :parameter length: The number of data points in the stream.
    :return: A list representing the generated data stream.
    """
    data_stream = []
    anomaly_labels=[]
    actual_anamolies=[]
    for t in range(length):
        # Regular pattern (e.g., daily cycles)
        regular_pattern = 50 * math.sin(2 * math.pi * t / 100)  # Regular cycle with period of 100

        # Seasonal pattern (e.g., yearly cycles)
        seasonal_pattern = 20 * math.sin(2 * math.pi * t / 365)  # Seasonal cycle with period of 365

        # Random noise
        noise = random.gauss(0, 5)  # Gaussian noise with mean=0, std=5

        # Combine components into a data point
        data_point = regular_pattern + seasonal_pattern + noise

        # Inject anomalies
        if random.random() > 0.95:  # 5% chance of anomaly
            data_point *= random.uniform(2, 5)  # Anomalous spike
            anomaly_labels.append(1)  # Label as anomaly
            if(t>=200):
                actual_anamolies.append(data_point)
        else:
            anomaly_labels.append(0)  # Label as normal


        # Append the generated data point to the stream
        data_stream.append(data_point)

    return data_stream,anomaly_labels,actual_anamolies