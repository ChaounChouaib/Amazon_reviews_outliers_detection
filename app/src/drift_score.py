from scipy.stats import wasserstein_distance
import numpy as np

class DriftScores :
    def __init__(self, data, outliers) -> None:
        self.data=data
        self.outliers=outliers # array of boolean values indicating the outliers detected by the autoencoder
        pass

    def wasserstein_distance_score(self):
        self.wass_distances = [wasserstein_distance(self.data[~self.outliers, i], self.data[self.outliers, i]) for i in range(self.data.shape[1])]
        self.avg_wasserstein_distance = sum(self.wass_distances) / len(self.wass_distances)
    
    def psi_score(self,columns) :
        num_bins = 10  
        bins = np.linspace(0, 1, num_bins + 1)
        self.feature_drift_psi = {col: calculate_psi(self.data[~self.outliers, i], self.data[self.outliers, i],bins) for i,col in zip(range(self.data.shape[1]),columns)}
    
    def run(self) :
        self.wasserstein_distance_score()
        self.psi_score()

    def report(self) :
        self.run()
        return {
            "wasserstein_distance_average_distance" : self.avg_wasserstein_distance,
            "psi_per_feature" : self.feature_drift_psi
            }

def calculate_psi(train, test, bins:list):
    # Divide both sets into bins and calculate PSI
    psi = 0
    for bin in bins:
        train_perc = len(train[(train >= bin) & (train < bin+1)]) / len(train)
        test_perc = len(test[(test >= bin) & (test < bin+1)]) / len(test)
        if test_perc > 0:
            psi += (train_perc - test_perc) * np.log(train_perc / test_perc)
    return psi