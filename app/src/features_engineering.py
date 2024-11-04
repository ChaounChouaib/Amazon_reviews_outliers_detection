import re
import string
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

text_features = ["review_title", "text", "product_title", "features"]
prefix_dict = {'review_title':"RTE_", 
                'text':'TE_',
                'product_title':'PTE_',
                'features':'FE_'} # This dictionnary will be used for adding prefix to the new features
n_components = 50 # Set desired dimension after embedding per feature
stop_words = set(stopwords.words('english')) # Beware to download stopwords else run : nltk.download('stopwords')
embedding_model = 'all-mpnet-base-v2' # Or for lighter processing use 'bert-base-nli-mean-tokens'

class TextProcessing :
    def __init__(self,data:pd.DataFrame,text_features = text_features) -> None:
        self.data = data
        self.text_features = text_features
    
    def pre_processing(self) :
        # Ensuring Str format
        for col in text_features : 
            self.data[col] = self.data[col].apply(str)
        
        # lower case
        for col in text_features :
            self.data[col] = self.data[col].str.lower()
        
        # Remove Urls
        for col in text_features :
            self.data[col] = self.data[col].replace(r'http\S+|www.\S+', '', regex=True)
        
        # Remove Punctuation
        for col in text_features :
            self.data[col] = self.data[col].str.translate(str.maketrans('', '', string.punctuation))
        
        # Removing stop words
        for col in text_features :
            self.data[col] = self.data[col].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
        
        # Removing special caracters
        for col in text_features :
            self.data[col] = self.data[col].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))  # Removes non-ASCII characters
        
    
    def lemmatizer(self) :
        # Lemmatization
        lemmatizer = WordNetLemmatizer()

        for col in text_features :
            self.data[col] = self.data[col].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))
    
    def embedding_process(self) :
        model = SentenceTransformer(embedding_model)
        for col in text_features :
            self.data[col] = model.encode(self.data[col].tolist()).tolist()

        self.data[text_features].to_pickle(f"data/embeddings/{embedding_model}.pkl")
    
    def run(self,retour=False) :
        self.pre_processing()
        self.lemmatizer()
        self.embedding_process()
        if retour :
            return self.data


class FeaturesEngineering :
    def __init__(self,data:pd.DataFrame) -> None:
        self.data = data
    
    def pca_embedding(self) :
        for col in text_features:
            # Step 1: Convert the list of embeddings into a 2D array
            embeddings_matrix = np.array(self.data[col].tolist())
            
            # Step 2: Apply PCA
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings_matrix)
            
            # Convert reduced embeddings back to list format for easy storage in DataFrame
            self.data[col] = list(reduced_embeddings)
    
    def explode_features(self) :
        """Transforming array in a cell into each value of the array in its own column"""
        for col in text_features :
            self.data = self.data.join(pd.DataFrame(self.data[col].tolist(), index=self.data.index).add_prefix(prefix_dict[col]))
            self.data.drop(col, axis =1 , inplace=True)
    
    def removing_ids(self) :
        # removing_unecessary features
        to_be_removed = ["parent_asin", # replaced by their embeddings
                     "main_category","images_of_product","user_rating_mean" # removed after corr map
                     ]
        to_be_removed = [col for col in to_be_removed if col in self.data.columns]
        self.data.drop(to_be_removed,axis=1,inplace=True)

    def data_rescaling(self) :
        scaler = RobustScaler()
        self.data = scaler.fit_transform(self.data)
    
    def run(self) :
        self.pca_embedding()
        self.explode_features()
        self.removing_ids()
        self.data_rescaling()     

    def train_test_val_split(self) :
        self.train_data, temp_data = train_test_split(self.data.reset_index(), test_size=0.3, random_state=42)
        self.validation_data, self.test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        return self.train_data, self.test_data, self.validation_data  