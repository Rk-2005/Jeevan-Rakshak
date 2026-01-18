"""
Water Quality Prediction Server
Provides API endpoints for water quality prediction using trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables to store trained model and scaler
trained_model = None
scaler = None
feature_columns = None

def preprocess_data(data):
    """Preprocess water quality data"""
    data.fillna(data.mean(), inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def create_complex_model(input_shape):
    """Create the deep learning model for predictions"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model_with_data(data):
    """Train model with provided data"""
    global trained_model, scaler, feature_columns
    
    try:
        # Preprocess data
        data = preprocess_data(data.copy())
        
        # Separate features and target
        if 'Potability' not in data.columns:
            return False, "Dataset must contain 'Potability' column"
        
        X = data.drop('Potability', axis=1)
        y = data['Potability']
        feature_columns = X.columns.tolist()
        
        # Remove rows with missing target values
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            return False, "No valid data after preprocessing"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=404
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        trained_model = create_complex_model((X_train_scaled.shape[1],))
        print("Training model... This may take a few minutes.")
        trained_model.fit(
            X_train_scaled, y_train, 
            epochs=50, 
            batch_size=64, 
            verbose=0,
            validation_split=0.1
        )
        
        # Evaluate model
        test_loss, test_accuracy = trained_model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Model trained successfully. Test Accuracy: {test_accuracy:.4f}")
        
        return True, f"Model trained with accuracy: {test_accuracy:.4f}"
    
    except Exception as e:
        return False, str(e)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_trained': trained_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict water potability from uploaded CSV file
    Expected CSV format: columns matching training data (without Potability column)
    """
    global trained_model, scaler, feature_columns
    
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400
        
        # Read CSV file
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # If no model is trained yet, train with this data
        if trained_model is None:
            success, message = train_model_with_data(data)
            if not success:
                return jsonify({'error': f'Failed to train model: {message}'}), 400
        
        # Prepare data for prediction
        data = preprocess_data(data.copy())
        
        # Handle feature columns
        if feature_columns is None:
            # If feature_columns not set, use all columns except 'Potability'
            if 'Potability' in data.columns:
                X = data.drop('Potability', axis=1)
            else:
                X = data
        else:
            # Use only the columns that were used during training
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                return jsonify({
                    'error': f'Missing columns: {missing_cols}. Expected columns: {feature_columns}'
                }), 400
            X = data[feature_columns]
        
        # Fill missing values
        X.fillna(X.mean(), inplace=True)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = trained_model.predict(X_scaled, verbose=0)
        potability_scores = predictions.flatten().tolist()
        
        # Convert to potability (1 for potable >= 0.5, 0 for non-potable < 0.5)
        potability_predictions = [float(score) for score in potability_scores]
        
        return jsonify({
            'potability_prediction': potability_predictions,
            'num_samples': len(potability_predictions),
            'potable_count': sum(1 for p in potability_predictions if p >= 0.5),
            'non_potable_count': sum(1 for p in potability_predictions if p < 0.5),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    return jsonify({
        'model_trained': trained_model is not None,
        'feature_columns': feature_columns,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Train model with uploaded CSV file
    CSV should contain 'Potability' column as target
    """
    global trained_model, scaler, feature_columns
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV format'}), 400
        
        # Read CSV file
        data = pd.read_csv(file)
        
        # Train model
        success, message = train_model_with_data(data)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'feature_columns': feature_columns,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': message}), 400
    
    except Exception as e:
        return jsonify({'error': f'Training error: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Water Quality Prediction Server")
    print("=" * 60)
    print("Server starting on http://127.0.0.1:5200")
    print("Endpoints:")
    print("  GET  /health         - Health check")
    print("  POST /predict        - Predict water potability from CSV")
    print("  POST /train          - Train model with CSV data")
    print("  GET  /model-info     - Get model information")
    print("=" * 60)
    
    app.run(host='127.0.0.1', port=5200, debug=False)
