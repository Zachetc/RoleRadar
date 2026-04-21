import requests

from data_input import BATCH_INPUT, SAMPLE_INPUT

BASE_URL = "http://127.0.0.1:5000"

single = requests.post(f"{BASE_URL}/predict", json=SAMPLE_INPUT, timeout=30)
print("Single prediction:")
print(single.json())

batch = requests.post(f"{BASE_URL}/predict_batch", json=BATCH_INPUT, timeout=30)
print("
Batch prediction:")
print(batch.json())
