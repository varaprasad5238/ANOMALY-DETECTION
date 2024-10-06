class CustomHoltWinters:
    def __init__(self, alpha, beta, gamma, seasonal_period):
        self.alpha = alpha  # Level smoothing parameter
        self.beta = beta    # Trend smoothing parameter
        self.gamma = gamma  # Seasonal smoothing parameter
        self.seasonal_period = seasonal_period
        self.level = None
        self.trend = None
        self.seasonals = None

    def initial_trend(self, data):
        trend = 0
        for i in range(self.seasonal_period):
            trend += (data[i + self.seasonal_period] - data[i]) / self.seasonal_period
        return trend / self.seasonal_period

    def initial_seasonal_components(self, data):
        seasonals = {}
        for i in range(self.seasonal_period):
            seasonals[i] = data[i] / (sum(data[:self.seasonal_period]) / self.seasonal_period)
        return seasonals

    def fit(self, data):
        self.level = sum(data[:self.seasonal_period]) / self.seasonal_period
        self.trend = self.initial_trend(data)
        self.seasonals = self.initial_seasonal_components(data)
        predictions = []
        for i in range(len(data)):
            m = i - self.seasonal_period
            if m >= 0:
                season = self.seasonals[m % self.seasonal_period]
                prediction = self.level + self.trend + season
            else:
                prediction = data[i]
            predictions.append(prediction)
            self.update(data[i], i)
        return predictions

    def update(self, data_point, t):
        last_level = self.level
        seasonal_index = t % self.seasonal_period
        self.level = self.alpha * (data_point / self.seasonals[seasonal_index]) + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        self.seasonals[seasonal_index] = self.gamma * (data_point / self.level) + (1 - self.gamma) * self.seasonals[seasonal_index]

    def predict(self, steps):
        forecasts = []
        for i in range(steps):
            season = self.seasonals[i % self.seasonal_period]
            forecast = self.level + self.trend * i + season
            forecasts.append(forecast)
        return forecasts
