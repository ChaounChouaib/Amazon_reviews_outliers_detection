import pandas as pd
import json


from src.data_preprocessing import read_pipeline
from src.data_transformation import DataTransformer
from src.drift_score import DriftScores
from src.features_engineering import FeaturesEngineering, TextProcessing
from src.model import AutoEncoder
from src.outlier_detection import Outliers

class Back:
    def __init__(self) -> None:
        pass
    
    def raw_data_processing(self,files_paths, review_conditions) :
        # reading and preprocessing data
        print("Reading data ...")
        df_preprocessed = read_pipeline(files_paths[0],files_paths[1],review_conditions)
        
        # Transforming the data
        print("transforming data ...")
        dt = DataTransformer(df_preprocessed)
        dt.run()

        # Text Processing
        print("Text Processing ...")
        tp = TextProcessing(dt.data)
        tp.run()

        # Feature engineering
        print("Features Engineering ...")
        fe=FeaturesEngineering(tp.data)
        fe.run()
        
        # saving data snapshot
        data_snapshot = f"data/processed/{files_paths[1].split('/')[-1].split('.')[0]}.csv"
        fe.data.to_csv(data_snapshot,index=False)

        # Model Training
        print("Training model ...")
        ae = AutoEncoder(fe.train_test_val_split())
        ae.train()
        return ae.model_id, ae.test_loss, data_snapshot

    def get_outliers(self,model_id, outliers_percentile, data_processed) :
        # loading data
        data = pd.read_csv(data_processed)
        fe=FeaturesEngineering(data)
        # loading model
        ae = AutoEncoder(fe.train_test_val_split())
        ae.load_model(model_id)

        # Detectin outliers
        ol = Outliers(ae.autoencoder,ae.test_data,outliers_percentile)
        ol.run()
        return ol.get_original_format().to_json()

    def report_outliers(self,model_id, outliers_percentile, data_processed) :
        # loading data
        data = pd.read_csv(data_processed)
        fe=FeaturesEngineering(data)
        # loading model
        ae = AutoEncoder(fe.train_test_val_split())
        ae.load_model(model_id)

        # Detectin outliers
        ol = Outliers(ae.autoencoder,ae.test_data,outliers_percentile)
        ol.run()

        # Drift Score
        ds = DriftScores(ae.test_data,ol.outliers)
        return ds.report()

    



