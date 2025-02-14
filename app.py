'''
This file is the API side of the FHE model. It is responsible for receiving 
encrypted data from the client, running the model, and returning the encrypted result to the client.
'''

from concrete.ml.deployment import FHEModelServer
from  flask import Flask, request
import base64

fhe_directory = '/tmp/fhe_client_server_files/'

server = FHEModelServer(path_dir=fhe_directory)
server.load()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    encrypted_data_b64 = request.data
    serialized_evaluation_keys_b64 = request.headers.get('Evaluation-Keys')

    encrypted_data = base64.b64decode(encrypted_data_b64)
    serialized_evaluation_keys = base64.b64decode(serialized_evaluation_keys_b64)
    
    encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
    encrypted_result_b64 = base64.b64encode(encrypted_result).decode('utf-8')

    return encrypted_result_b64

if __name__ == '__main__':
    app.run(port=5001,debug=True)