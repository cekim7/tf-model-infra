import requests
import json
import numpy as np

def send_prediction_request(api_url, inputs):
    headers = {'Content-Type': 'application/json'}

    # Ensure inputs are in the correct format (list of lists) and convert from NumPy
    if isinstance(inputs, np.ndarray):
      data = {"inputs": inputs.tolist()}
    else:
      data = {"inputs": inputs}

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

if __name__ == "__main__":
    api_url = "http://localhost:8501/predict" # Replace with your service URL if not running locally
    inputs = np.array([[1], [2], [3]])  # Example NumPy input
    # inputs = [[1], [2], [3]] # Example list of lists input

    predictions = send_prediction_request(api_url, inputs)

    if predictions:
      print("Predictions:")
      print(json.dumps(predictions, indent=2))