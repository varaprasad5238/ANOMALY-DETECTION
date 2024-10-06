import matplotlib.pyplot as plt


class RealTimePlot:
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.fig, self.ax = plt.subplots()
        self.data_line, = self.ax.plot([], [], label="Data Stream")
        self.anomaly_line, = self.ax.plot([], [], 'ro', label="Anomalies")  # Red dots for anomalies
        self.ax.legend()
        self.ax.set_title("Real-Time Data Stream with Anomaly Detection")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")

    def initialize_plot(self):
        self.data_line.set_data([], [])
        self.anomaly_line.set_data([], [])
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(-100, 100)  # Modify according to expected data range
        return self.data_line, self.anomaly_line

    def update_plot(self, x_data, y_data, anomalies):
        # Update the data stream line
        self.data_line.set_data(x_data, y_data)

        # Update anomaly points
        anomaly_x = [x_data[i] for i in range(len(x_data)) if anomalies[i]]
        anomaly_y = [y_data[i] for i in range(len(y_data)) if anomalies[i]]
        self.anomaly_line.set_data(anomaly_x, anomaly_y)

        # Adjust x-axis window to scroll as time progresses
        if len(x_data) > self.window_size:
            self.ax.set_xlim(max(0, x_data[-self.window_size]), x_data[-1])

        self.ax.set_ylim(min(y_data) - 10, max(y_data) + 10)  # Adjust based on the range of data
        plt.pause(0.01)  # Pause to update the plot in real-time

    def show_plot(self):
        plt.show()