'''
This script is used to send encrypted data to the REST API and receive the encrypted response.
'''

from concrete.ml.deployment import FHEModelClient
import numpy as np
import requests, base64

fhe_directory = '/tmp/fhe_client_server_files/'

client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

X_new = [[0.7328857 , 0.55031411, 0.26744947, 0.34693709, 0.76240011,
        0.54259767, 0.66891019, 0.66172162, 0.70467078, 0.12846003,
        0.57701154, 0.91836001, 0.36661011, 0.55454585, 0.21517037,
        0.06911301, 0.85830442, 0.67499044, 0.6111592 , 0.1908567 ]]

#X_new = np.random.rand(1, 20)

encrypted_data = client.quantize_encrypt_serialize(X_new)


encrypted_data_b64 = base64.b64encode(encrypted_data).decode('utf-8')
serialized_evaluation_keys_b64 = base64.b64encode(serialized_evaluation_keys).decode('utf-8')

encrypted_response_b64 = requests.post(
    "http://127.0.0.1:5001/predict", 
    data=encrypted_data_b64, 
    headers={"Content-Type": "application/octet-stream", "Evaluation-Keys": serialized_evaluation_keys_b64})

encrypted_response = base64.b64decode(encrypted_response_b64.content)

result = client.deserialize_decrypt_dequantize(encrypted_response)

print(result)