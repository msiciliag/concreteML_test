'''
This file is the API side of the FHE model. It is responsible for receiving 
encrypted data from the client, running the model, and returning the encrypted result to the client.
'''

from concrete.ml.deployment import FHEModelServer
from flask import Flask, request

fhe_directory = '/tmp/fhe_client_server_files/'

server = FHEModelServer(path_dir=fhe_directory)
server.load()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    encrypted_data = request.data
    serialized_evaluation_keys = request.headers.get('Evaluation-Keys').encode('utf-8')
    
    encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
    
    return encrypted_result, 200, {'Content-Type': 'application/octet-stream'}

if __name__ == '__main__':
    app.run(port=5001, debug=True)