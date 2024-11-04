import pandas as pd
from sklearn.preprocessing import LabelEncoder

def labelencoder(s:pd.Series) :
    # Initialize the label encoder
    label_encoder = LabelEncoder() 

    # Apply label encoding
    return label_encoder.fit_transform(s)

class DataTransformer:
    def __init__(self, data : pd.DataFrame) -> None:
        self.data : data

    def handeling_missing_values(self) :
        # if the product info is not provided we drop the line
        self.data.drop(self.data[self.data.product_title.isna()].index,inplace=True)
        # for text review data we will fill using ''
        self.data.fillna(value={'review_title' : '', 'text':''},inplace=True)

    def categorical_encoding(self) :
        self.data.verified_purchase = labelencoder(self.data.verified_purchase)
        self.data.main_category = labelencoder(self.data.main_category)
        self.data.verified_purchase = labelencoder(self.data.verified_purchase)
        self.data.store = labelencoder(self.data.store)
    
    def timestamp_transforming(self) :
        self.data.timestamp = pd.to_datetime(self.data.timestamp)
        self.data["year"] = self.data.timestamp.dt.year
        self.data["day"] = self.data.timestamp.dt.day_of_year
        self.data["hour"] = self.data.timestamp.dt.hour
        self.data.drop("timestamp", axis=1, inplace=True)
    
    def user_id_transforming(self) :
        user_aggregated = self.data.groupby("user_id").agg({"rating" : ["count","mean"]})
        user_aggregated.columns = ['_'.join(col) for col in user_aggregated.columns]
        user_aggregated.reset_index(inplace=True)
        self.data = self.data.merge(user_aggregated,on='user_id',how='left')
        self.data.rename(columns={'rating_count':'user_rating_count','rating_mean':'user_rating_mean'},inplace=True)
        self.data.drop('user_id',axis=1,inplace=True)
    
    def run(self,retour = False) :
        self.handeling_missing_values()
        self.categorical_encoding()
        self.timestamp_transforming()
        self.user_id_transforming
        if retour :
            return self.data
