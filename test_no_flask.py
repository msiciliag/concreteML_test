'''
This script tests the FHE concrete.ml.deployment module without using REST API requests.
Tested on 14 feb 2025:
X_new = [[0.7328857 , 0.55031411, 0.26744947, 0.34693709, 0.76240011,
        0.54259767, 0.66891019, 0.66172162, 0.70467078, 0.12846003,
        0.57701154, 0.91836001, 0.36661011, 0.55454585, 0.21517037,
        0.06911301, 0.85830442, 0.67499044, 0.6111592 , 0.1908567 ]]
result = [[0.52161591 0.47838409]]
'''

from concrete.ml.deployment import FHEModelClient, FHEModelServer
import numpy as np

fhe_directory = '/tmp/fhe_client_server_files/'

client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
server = FHEModelServer(path_dir=fhe_directory)
server.load()

X_new = np.random.rand(1, 20)

encrypted_data = client.quantize_encrypt_serialize(X_new)
encrypted_result = server.run(encrypted_data, client.get_serialized_evaluation_keys())
result = client.deserialize_decrypt_dequantize(encrypted_result)

print(result)
