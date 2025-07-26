# train_model.py - Run this once to train and save your model
from plane_crash_predictor import PlaneCrashSurvivalPredictor

print("Training the Plane Crash Survival Prediction Model...")
print("="*60)

# Initialize predictor
predictor = PlaneCrashSurvivalPredictor()

# Train the model using unbalanced data (better test accuracy)
try:
    train_accuracy = predictor.train('train.csv')
    
    # Save the trained model
    predictor.save_model('plane_crash_model.pkl')
    
    print(f"\n✅ Model training completed successfully!")
    print(f"✅ Training accuracy: {train_accuracy:.4f}")
    print(f"✅ Model saved as 'plane_crash_model.pkl'")
    
    # Test a sample prediction
    print(f"\n🧪 Testing sample prediction...")
    prediction, probabilities = predictor.predict_single(
        flight_phase='Takeoff',
        water_body_type='Ocean',
        weather_condition='Clear',
        day_period='Day',
        cause_category='Human Error',
        aboard=150
    )
    
    print(f"Sample prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
    
    # Show feature importance
    print(f"\n📊 Feature Importance:")
    importance_df = predictor.get_feature_importance()
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n🚀 Ready to run the web application!")
    print(f"Run: python app.py")
    
except FileNotFoundError:
    print(f"❌ Error: 'train.csv' file not found!")
    print(f"Please make sure 'train.csv' is in the current directory.")
    
except Exception as e:
    print(f"❌ Error during training: {e}")