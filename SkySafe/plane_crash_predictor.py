import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

class PlaneCrashSurvivalPredictor:
    """
    Plane Crash Survival Prediction Model for Web Application
    Uses unbalanced data for better test accuracy
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_columns = ['FlightPhase', 'WaterBodyType', 'WeatherCondition', 'DayPeriod', 'CauseCategory', 'Aboard']
        self.categorical_cols = ['FlightPhase', 'WaterBodyType', 'WeatherCondition', 'DayPeriod', 'CauseCategory']
        self.quantitative_cols = ['Aboard']
        self.target = 'SurvivalSeverity'
        self.is_trained = False
        
        # Best parameters found from your analysis
        self.best_params = {
            'criterion': 'gini',
            'max_depth': 15,
            'min_samples_leaf': 2,
            'min_samples_split': 10,
            'random_state': 42
        }
    
    def handle_missing_values(self, X, fit_on=None):
        """Handle missing values in features"""
        X_clean = X.copy()
        
        # For categorical columns - fill with mode
        for col in self.categorical_cols:
            if col in X_clean.columns and X_clean[col].isnull().sum() > 0:
                if fit_on is not None and col in fit_on.columns:
                    mode_val = fit_on[col].mode()[0] if not fit_on[col].mode().empty else 'Unknown'
                else:
                    mode_val = X_clean[col].mode()[0] if not X_clean[col].mode().empty else 'Unknown'
                X_clean[col].fillna(mode_val, inplace=True)
        
        # For quantitative columns - fill with median
        for col in self.quantitative_cols:
            if col in X_clean.columns and X_clean[col].isnull().sum() > 0:
                if fit_on is not None and col in fit_on.columns:
                    median_val = fit_on[col].median()
                else:
                    median_val = X_clean[col].median()
                X_clean[col].fillna(median_val, inplace=True)
        
        return X_clean
    
    def encode_features(self, X, fit=False):
        """Encode categorical features"""
        X_encoded = X.copy()
        
        for col in self.categorical_cols:
            if col in X_encoded.columns:
                if fit:
                    # Fit encoder on training data
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Transform using existing encoder
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        X_col = X[col].astype(str)
                        
                        # Handle unseen categories
                        mask = X_col.isin(le.classes_)
                        X_encoded[col] = 0  # Default for unseen categories
                        X_encoded.loc[mask, col] = le.transform(X_col[mask])
                    else:
                        # If no encoder exists, assign 0
                        X_encoded[col] = 0
        
        return X_encoded
    
    def train(self, train_csv_path='train.csv'):
        """Train the model using unbalanced data for better test accuracy"""
        print("Loading training data...")
        try:
            train_df = pd.read_csv(train_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Training file {train_csv_path} not found")
        
        print(f"Training data shape: {train_df.shape}")
        
        # Check if required columns exist
        missing_cols = [col for col in self.feature_columns + [self.target] if col not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in training data: {missing_cols}")
        
        # Prepare features and target
        X_train = train_df[self.feature_columns].copy()
        y_train = train_df[self.target].copy()
        
        print(f"Original class distribution:")
        print(y_train.value_counts().sort_index())
        
        # Handle missing values
        X_train_clean = self.handle_missing_values(X_train)
        
        # Encode categorical features
        X_train_encoded = self.encode_features(X_train_clean, fit=True)
        
        # Encode target if categorical
        if y_train.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y_train_encoded = self.target_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train.copy()
        
        # Train the model (using UNBALANCED data for better test accuracy)
        print("\nTraining Decision Tree model with unbalanced data...")
        self.model = DecisionTreeClassifier(**self.best_params)
        self.model.fit(X_train_encoded.values, y_train_encoded)
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train_encoded.values)
        from sklearn.metrics import accuracy_score
        train_accuracy = accuracy_score(y_train_encoded, train_predictions)
        
        print(f"Training completed successfully!")
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        self.is_trained = True
        return train_accuracy
    
    def predict_single(self, flight_phase, water_body_type, weather_condition, 
                      day_period, cause_category, aboard):
        """
        Make prediction for a single instance
        
        Parameters:
        - flight_phase: str (e.g., 'Takeoff', 'Cruise', 'Landing', etc.)
        - water_body_type: str (e.g., 'Ocean', 'Lake', 'River', etc.)
        - weather_condition: str (e.g., 'Clear', 'Rain', 'Snow', etc.)
        - day_period: str (e.g., 'Day', 'Night', 'Dawn', etc.)
        - cause_category: str (e.g., 'Human Error', 'Mechanical', etc.)
        - aboard: int (number of people aboard)
        
        Returns:
        - prediction: str (survival severity category)
        - probability: dict (probability for each class)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'FlightPhase': [flight_phase],
            'WaterBodyType': [water_body_type],
            'WeatherCondition': [weather_condition],
            'DayPeriod': [day_period],
            'CauseCategory': [cause_category],
            'Aboard': [aboard]
        })
        
        # Handle missing values
        input_clean = self.handle_missing_values(input_data)
        
        # Encode features
        input_encoded = self.encode_features(input_clean, fit=False)
        
        # Make prediction
        prediction_encoded = self.model.predict(input_encoded.values)[0]
        probabilities = self.model.predict_proba(input_encoded.values)[0]
        
        # Decode prediction if target was encoded
        if self.target_encoder is not None:
            prediction = self.target_encoder.inverse_transform([prediction_encoded])[0]
            class_labels = self.target_encoder.classes_
        else:
            prediction = prediction_encoded
            class_labels = sorted(set(self.model.classes_))
        
        # Create probability dictionary
        prob_dict = {str(label): float(prob) for label, prob in zip(class_labels, probabilities)}
        
        return prediction, prob_dict
    
    def predict_batch(self, input_df):
        """
        Make predictions for multiple instances
        
        Parameters:
        - input_df: DataFrame with columns matching feature_columns
        
        Returns:
        - predictions: list of predictions
        - probabilities: list of probability dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Ensure all required columns are present
        missing_cols = [col for col in self.feature_columns if col not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Handle missing values
        input_clean = self.handle_missing_values(input_df)
        
        # Encode features
        input_encoded = self.encode_features(input_clean, fit=False)
        
        # Make predictions
        predictions_encoded = self.model.predict(input_encoded.values)
        probabilities = self.model.predict_proba(input_encoded.values)
        
        # Decode predictions if target was encoded
        if self.target_encoder is not None:
            predictions = self.target_encoder.inverse_transform(predictions_encoded)
            class_labels = self.target_encoder.classes_
        else:
            predictions = predictions_encoded
            class_labels = sorted(set(self.model.classes_))
        
        # Create probability dictionaries
        prob_dicts = []
        for prob_array in probabilities:
            prob_dict = {str(label): float(prob) for label, prob in zip(class_labels, prob_array)}
            prob_dicts.append(prob_dict)
        
        return predictions.tolist(), prob_dicts
    
    def get_feature_importance(self):
        """Get feature importance for the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath='plane_crash_model.pkl'):
        """Save the trained model and encoders"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns,
            'categorical_cols': self.categorical_cols,
            'quantitative_cols': self.quantitative_cols,
            'target': self.target,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='plane_crash_model.pkl'):
        """Load a pre-trained model and encoders"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.target_encoder = model_data['target_encoder']
            self.feature_columns = model_data['feature_columns']
            self.categorical_cols = model_data['categorical_cols']
            self.quantitative_cols = model_data['quantitative_cols']
            self.target = model_data['target']
            self.best_params = model_data['best_params']
            
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {filepath} not found")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")


# Example usage for web application
if __name__ == "__main__":
    # Initialize and train the model
    predictor = PlaneCrashSurvivalPredictor()
    
    # Train the model (make sure train.csv is available)
    try:
        train_accuracy = predictor.train('train.csv')
        
        # Save the model for web application use
        predictor.save_model('plane_crash_model.pkl')
        
        # Example prediction
        prediction, probabilities = predictor.predict_single(
            flight_phase='Takeoff',
            water_body_type='Ocean',
            weather_condition='Clear',
            day_period='Day',
            cause_category='Human Error',
            aboard=150
        )
        
        print(f"\nExample prediction:")
        print(f"Predicted survival severity: {prediction}")
        print(f"Probabilities: {probabilities}")
        
        # Show feature importance
        print(f"\nFeature importance:")
        print(predictor.get_feature_importance())
        
    except Exception as e:
        print(f"Error: {e}")


# Web application integration functions
def create_web_predictor():
    """Create and return a predictor instance for web app"""
    predictor = PlaneCrashSurvivalPredictor()
    try:
        predictor.load_model('plane_crash_model.pkl')
        return predictor
    except:
        # If model doesn't exist, train a new one
        predictor.train('train.csv')
        predictor.save_model('plane_crash_model.pkl')
        return predictor

def web_predict(flight_phase, water_body_type, weather_condition, 
                day_period, cause_category, aboard):
    """
    Simple function for web application to make predictions
    
    Usage in Flask/FastAPI:
    prediction, probabilities = web_predict(
        flight_phase, water_body_type, weather_condition,
        day_period, cause_category, aboard
    )
    """
    predictor = create_web_predictor()
    return predictor.predict_single(
        flight_phase, water_body_type, weather_condition,
        day_period, cause_category, aboard
    )