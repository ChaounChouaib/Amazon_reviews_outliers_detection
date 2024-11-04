import pandas as pd
import jsonlines
from tqdm import tqdm

# reading json data
def read_json_data(meta_file_path, review_file_path, review_filter = False, review_conditions=dict()) :
    df_meta = pd.read_json(meta_file_path, lines=True)
    if review_filter :
        df_reviews = read_json_with_conditions(review_file_path,review_conditions)
    else :
        df_reviews = pd.read_json(review_file_path, lines=True)
    return df_meta, df_reviews


def read_json_with_conditions(review_file_path,review_conditions):
    # Read and filter data
    filtered_data = []
    with jsonlines.open(review_file_path) as reader:
        for obj in tqdm(reader):
            condition = True
            for key in review_conditions :
                if key == 'timestamp' :
                    if pd.to_datetime(obj["timestamp"], errors='coerce', unit='ms').year not in review_conditions[key] :
                        condition = False
                if obj[key] not in review_conditions[key]:
                    condition = False
            if condition : 
                filtered_data.append(obj)
    return pd.DataFrame(data = filtered_data)

def meta_data_preprocessing(df_meta) :
    df_meta.drop(["price","bought_together","description","videos","details","categories","asin"], axis=1, inplace=True)
    df_meta.images = df_meta.images.apply(len) # transforming images feature
    df_meta.features = df_meta.features.apply(lambda x:' '.join(x))
    return df_meta

def review_data_preprocessing(df_reviews) :
    df_reviews.timestamp = df_reviews.timestamp.apply(lambda timestamp : pd.to_datetime(timestamp, errors='coerce', unit='ms') )
    df_reviews.images = df_reviews.images.apply(len)
    return df_reviews

def concate(df_meta,df_reviews) :
    df = pd.merge(
                    df_meta.rename(columns={'images' : 'images_by_user'}), 
                    df_reviews.rename(columns={'images' : 'images_of_product'}),
                    on='parent_asin'
                    )
    df.drop_duplicates(inplace=True)
    return df

def read_pipeline(meta_file_path, review_file_path, review_conditions=dict()) :
    if len(review_conditions) : 
        review_filter = True
    else :
        review_filter = False

    # Reading data
    df_meta , df_reviews = read_json_data(
        meta_file_path, review_file_path, 
        review_filter = review_filter, review_conditions=review_conditions
    )
    # Preprocessing data
    df_meta = meta_data_preprocessing(df_meta)
    df_reviews = review_data_preprocessing(df_reviews)
    # join data
    return concate(df_meta, df_reviews)

if __name__=="__main__" :
    meta_file_path = 'path' # Put json meta data file 
    review_file_path = 'path' # Put json reviews file
    review_conditions = {'timestamp' : [2018,2019,2020]}
    df_meta , df_reviews = read_json_data(
        meta_file_path, review_file_path, 
        review_filter = True, review_conditions=review_conditions
    )
    df_meta = meta_data_preprocessing(df_meta)
    df_reviews = review_data_preprocessing(df_reviews)
    df = concate(df_meta, df_reviews)
    print(df.shape)
    print(df.head())