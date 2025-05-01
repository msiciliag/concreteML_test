'''
This script tests the FHE concrete.ml.deployment module without using REST API requests.
'''

from concrete.ml.deployment import FHEModelClient, FHEModelServer
import numpy as np

fhe_directory = '/tmp/fhe_client_server_files/'

client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

server = FHEModelServer(path_dir=fhe_directory)
server.load()

X_new = [[0.0, 1.0, 1.0, 1.0, 0.35, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.1, 0.0, 1.0, 1.0, 0.7]]

encrypted_data = client.quantize_encrypt_serialize(X_new)
encrypted_result = server.run(encrypted_data, client.get_serialized_evaluation_keys())
result = client.deserialize_decrypt_dequantize(encrypted_result)

print(result)
