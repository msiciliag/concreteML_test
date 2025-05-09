'''
This file is the API side of the FHE model. It is responsible for receiving 
encrypted data from the client, the evaluation keys, running the model, and returning the encrypted result to the client.
'''

from concrete.ml.deployment import FHEModelServer
from flask import Flask, request

FHE_DIRECTORY = '/tmp/fhe_client_server_files/'

server = FHEModelServer(path_dir=FHE_DIRECTORY)
server.load()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    encrypted_data_file = request.files.get('encrypted_data')
    evaluation_keys_file = request.files.get('evaluation_keys')

    if not encrypted_data_file or not evaluation_keys_file:
        return "Missing files in the request", 400

    encrypted_data = encrypted_data_file.read()
    serialized_evaluation_keys = evaluation_keys_file.read()

    encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)

    return encrypted_result, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    app.run(port=5005, debug=True)