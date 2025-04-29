
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_harmony_model(input_dim, output_dim, hidden_layers=[128, 64, 32]):
    """Build a deep learning model for LifeHarmony recommendations."""
    inputs = layers.Input(shape=(input_dim,))
    
    # Normalize inputs
    x = layers.BatchNormalization()(inputs)
    
    # Hidden layers
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    
    # Output layer with sigmoid activation for multi-label classification
    outputs = layers.Dense(output_dim, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(dataset_path):
    """Prepare the dataset for deep learning."""
    # Load the generated dataset
    data = pd.read_excel(dataset_path)
    
    # Create mapping dictionaries
    marital_status_mapping = {"Single": 0, "Married": 1}
    occupation_mapping = {"Full-time": 0, "Part-time": 1, "Freelancer": 2, "Student": 3, "Unemployed": 4}
    personality_mapping = {"Extrovert": 0, "Introvert": 1, "Ambivert": 2}
    hobby_mapping = {"Exercise": 0, "Reading": 1, "Writing": 2, "Art": 3, "Socializing": 4}
    priority_mapping = {"Low": 0, "Medium": 1, "High": 2}
    
    # Create features array
    X = []
    for _, row in data.iterrows():
        features = [
            int(row['Age']),
            marital_status_mapping[row['Marital Status']],
            occupation_mapping[row['Occupation']],
            int(row['Budget']),
            personality_mapping[row['Personality']],
            hobby_mapping[row['Hobbies']],
        ]
        
        # Add priorities for each life domain
        life_features = ["Career", "Financial", "Spiritual", "Physical", "Intellectual", "Family", "Social", "Fun"]
        for feature in life_features:
            features.append(priority_mapping[row[f'{feature}_Priority']])
        
        X.append(features)
    
    X = np.array(X)
    
    # Extract and encode recommendations
    all_recommendations = set()
    for _, row in data.iterrows():
        all_recommendations.update(eval(row['Recommendations']))
    
    unique_recommendations = list(all_recommendations)
    recommendation_to_index = {rec: i for i, rec in enumerate(unique_recommendations)}
    
    # Create multi-hot encoding for recommendations
    y = np.zeros((len(data), len(unique_recommendations)))
    for i, row in enumerate(data.iterrows()):
        recs = eval(row[1]['Recommendations'])
        for rec in recs:
            y[i, recommendation_to_index[rec]] = 1
    
    # Normalize numerical features
    scaler = StandardScaler()
    X[:, [0, 3]] = scaler.fit_transform(X[:, [0, 3]])  # Scale Age and Budget
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, unique_recommendations, scaler

def train_model(dataset_path="generated_datasets/4_generated_dataset_with_recommendations.xlsx", model_save_path="harmony_deep_model"):
    """Train deep learning model and save it."""
    X_train, X_val, y_train, y_val, unique_recommendations, scaler = prepare_data(dataset_path)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = build_harmony_model(input_dim, output_dim)
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save the model
    model.save(f"{model_save_path}.h5")
    
    # Save the recommendation mapping and scaler
    with open(f"{model_save_path}_recommendations.pkl", "wb") as f:
        pickle.dump(unique_recommendations, f)
    
    with open(f"{model_save_path}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to {model_save_path}.h5")
    print(f"Recommendations saved to {model_save_path}_recommendations.pkl")
    print(f"Scaler saved to {model_save_path}_scaler.pkl")
    
    return model, history, unique_recommendations

def get_recommendations(user_features, model_path="harmony_deep_model.h5",
                       recommendations_path="harmony_deep_model_recommendations.pkl",
                       scaler_path="harmony_deep_model_scaler.pkl",
                       threshold=0.3, top_k=10):
    """Get recommendations for a user using the deep learning model."""
    # Load model and recommendations
    model = tf.keras.models.load_model(model_path)
    
    with open(recommendations_path, "rb") as f:
        unique_recommendations = pickle.load(f)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Apply same scaling to age and budget
    user_features = np.array(user_features).reshape(1, -1)
    user_features[:, [0, 3]] = scaler.transform(user_features[:, [0, 3]])
    
    # Get predictions
    predictions = model.predict(user_features)[0]
    
    # Method 1: Threshold-based selection
    recommended_indices = np.where(predictions > threshold)[0]
    
    # Method 2: If too few or too many recommendations, use top-k
    if len(recommended_indices) < 3 or len(recommended_indices) > top_k:
        recommended_indices = np.argsort(predictions)[-top_k:]
    
    # Get recommendations
    recommendations = [unique_recommendations[i] for i in recommended_indices]
    
    return recommendations

if __name__ == "__main__":
    # Train the model when this script is run directly
    train_model()
