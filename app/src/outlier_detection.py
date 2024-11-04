import numpy as np
import pandas as pd

class Outliers:
    def __init__(self,autoencoder,test_data,percentile = 99.9) -> None:
        self.autoencoder = autoencoder
        self.test_data = test_data
        self.percentile = percentile
    
    def reconstructed_error(self) :
        reconstructed_data = self.autoencoder.predict(self.test_data)
        self.reconstruction_errors = np.mean(np.square(self.test_data - reconstructed_data), axis=1)  # MSE per sample
    
    def threshold_outliers(self) :
        self.threshold = np.percentile(self.reconstruction_errors, self.percentile)
        self.outliers = self.reconstruction_errors > self.threshold
    
    def run(self):
        self.reconstructed_error()
        self.threshold_outliers()
    
    def get_original_format(self,df,original_index) :
        outliers_index = list(pd.Series(original_index)[self.outliers].values)
        return df.iloc[outliers_index]
        
