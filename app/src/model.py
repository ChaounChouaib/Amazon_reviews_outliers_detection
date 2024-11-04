from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
import json
from datetime import datetime

class AutoEncoder:
    def __init__(self,train_data,validation_data,test_data) -> None:
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

    def auto_encoder(self) :
        # Step 2: Define the Autoencoder Model
        input_dim = self.train_data.shape[1]
        encoding_dim = 10  # to be adjusted (since input dim > 200 reducing it to 10 will be good, depends on pca_embedding)

        autoencoder = Sequential([
            # Encoder
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dense(encoding_dim, activation='relu'),  # Encoded layer

            # Decoder
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(input_dim, activation='sigmoid')  # Output layer
        ])

        # Compile the model
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # Step 3: Train the Autoencoder
        self.history = autoencoder.fit(
            self.train_data,
            self.train_data,  # Target is the same as input for autoencoders
            epochs=50,
            batch_size=32,
            validation_data=(self.validation_data, self.validation_data)
        )

        # Step 4: Evaluate on Test Set
        self.test_loss = autoencoder.evaluate(self.test_data, self.test_data)
    
    def save_model(self) :
        current_time = datetime.now()
        self.model_id = f"{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"
        # Save model architecture as JSON
        model_json = self.autoencoder.to_json()
        with open(f"data/models/autoencoder_model_{self.model_id}.json", "w") as json_file:
            json_file.write(model_json)

        # Save model weights in a pickle file
        model_weights = self.autoencoder.get_weights()
        with open(f"data/models/autoencoder_weights_{self.model_id}.pkl", "wb") as file:
            pickle.dump(model_weights, file)

        # 2. Save Training History as JSON or Pickle
        # Convert history.history dict to JSON
        with open(f"data/models/training_history_{self.model_id}.json", "w") as file:
            json.dump(self.history.history, file)
    
    def load_model(self,model_id) :
        self.model_id = model_id
        # Load model architecture
        with open(f"data/models/autoencoder_model_{self.model_id}.json", "r") as json_file:
            loaded_model_json = json_file.read()
        self.autoencoder = model_from_json(loaded_model_json)

        # Load model weights
        with open(f"data/models/autoencoder_weights_{self.model_id}.pkl", "rb") as file:
            model_weights = pickle.load(file)
        self.autoencoder.set_weights(model_weights)

        # Compile the model (required after loading)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # Load training history
        with open(f"data/models/training_history_{self.model_id}.json", "r") as file:
            self.history = json.load(file)
    
    def train(self) :
        self.auto_encoder()
        self.save_model()