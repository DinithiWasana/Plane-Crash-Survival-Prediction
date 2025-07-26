# api.server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from plane_crash_predictor import web_predict, create_web_predictor

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Load model once when server starts
predictor = create_web_predictor()

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction, probabilities = web_predict(
            data['flight_phase'],
            data['water_body_type'],
            data['weather_condition'],
            data['day_period'],
            data['cause_category'],
            int(data['aboard'])
        )
        confidence = max(probabilities.values()) * 100

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': confidence,
            'input_data': data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'success': True, 'model_loaded': predictor.is_trained})

if __name__ == '__main__':
    app.run(port=5000)
