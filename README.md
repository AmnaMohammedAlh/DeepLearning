# Deep Learning Model:
Core machine learning infrastructure for the LifeHarmony recommender system. 

## 1. Model Architecture 
pythondef build_harmony_model(input_dim, output_dim, hidden_layers=[128, 64, 32]):

Architecture: A feed-forward neural network with configurable hidden layers

Input: User attributes and domain priorities (14 features total)
Processing:

- Input normalization with BatchNormalization
- Multiple dense layers with ReLU activation
- Dropout (30%) to prevent overfitting


Output: Multi-label classification for recommendations using sigmoid activation
Training: Uses Adam optimizer and binary cross-entropy loss function

This architecture is well-suited for a recommendation system as it allows the model to learn complex relationships between user attributes and appropriate recommendations.

## 2. Data Preparation (prepare_data)

Input Processing:

- Converts categorical features (marital status, occupation, etc.) to numerical values
- Creates a fixed-length feature vector for each user
- Includes both user attributes and life domain priorities


Output Processing:

- Creates a multi-hot encoding for recommendations
- Each recommendation is treated as a binary classification problem


Feature Scaling:

- Uses StandardScaler to normalize age and budget
- Preserves categorical encodings for other features


Data Splitting:

- 80% training, 20% validation with fixed random seed for reproducibility



## 3. Model Training (train_model)

Training Process:

- Uses early stopping to prevent overfitting
- Monitors validation loss to determine optimal training duration
- Uses batch size of 32 and maximum 100 epochs


Model Persistence:

- Saves the trained model in H5 format
- Stores recommendation mappings and feature scaler in pickle files
- This enables later recommendation generation without retraining



## 4. Recommendation Generation (get_recommendations)
Input Processing:

- Takes a feature vector representing a user
- Applies the same scaling used during training


Prediction Strategy:

- Generates probability scores for all possible recommendations
- Uses a hybrid approach for recommendation selection:

- First tries threshold-based selection (recommendations with probability > 0.3) If too few or too many recommendations, falls back to top-k selection




Output:

- Returns a list of relevant recommendations



## Integration with User Interfaces
This model module is designed to work with different interfaces:

Notebook Interface: The training process can be run in a Jupyter notebook
Streamlit Interface: The model can generate recommendations for the web UI
Terminal Interface: The model works with the command-line interface we created



