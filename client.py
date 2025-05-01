'''
This script is used to send encrypted data to the REST API and receive the encrypted response.
'''

from concrete.ml.deployment import FHEModelClient
import numpy as np
import requests

fhe_directory = '/tmp/fhe_client_server_files/'

client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

X_new = [[0.0, 1.0, 1.0, 1.0, 0.35, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.1, 0.0, 1.0, 1.0, 0.7]]

encrypted_data = client.quantize_encrypt_serialize(X_new)

encrypted_response = requests.post(
    "http://127.0.0.1:5001/predict", 
    data=encrypted_data, 
    headers={
        "Content-Type": "application/octet-stream", 
        "Evaluation-Keys": serialized_evaluation_keys
    }
)

result = client.deserialize_decrypt_dequantize(encrypted_response.content)

print(result)