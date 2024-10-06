def ensemble_voting(*models_results):
    """
    Simple majority voting ensemble. If the majority of models consider a data point as an anomaly, it is flagged as such.
    :parameter models_results: List of booleans from different models indicating whether a data point is an anomaly.
    :return: Boolean indicating if the data point is considered an anomaly by the ensemble.
    """
    return sum(models_results) >= len(models_results)/2