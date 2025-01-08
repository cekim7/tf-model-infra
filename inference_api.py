from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import prometheus_client
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

# Load the model here
model = tf.keras.models.load_model('./model.keras')

# Prometheus metrics
request_count = prometheus_client.Counter('model_api_request_total', 'Total number of requests')
prediction_count = prometheus_client.Counter('model_api_prediction_total', 'Total number of predictions')
prediction_histogram = prometheus_client.Histogram('model_api_prediction_latency', 'Prediction latency')

@app.route("/predict2", methods=["POST"])
def predict():
    request_count.inc()
    try:
        data = request.get_json()
        input_data = np.array(data['inputs'])
        with prediction_histogram.time():
            predictions = model.predict(input_data)
        prediction_count.inc()
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Add prometheus wsgi to route the metrics
    app_dispatch = DispatcherMiddleware(app, {
        '/metrics': make_wsgi_app()
    })
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8501, app_dispatch)