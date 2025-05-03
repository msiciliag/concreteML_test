'''
This script is used to send encrypted data to the REST API and receive the encrypted response.
'''

from concrete.ml.deployment import FHEModelClient
import numpy as np
import requests

fhe_directory = '/tmp/fhe_client_server_files/'

client = FHEModelClient(path_dir=fhe_directory, key_dir="/tmp/keys_client")
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

X_new = np.array([[
    1,  # HighBP (presión alta)
    0,  # HighChol (sin colesterol alto)
    1,  # CholCheck (chequeo de colesterol realizado)
    25, # BMI (índice de masa corporal)
    0,  # Smoker (no fumador)
    0,  # Stroke (sin historial de derrame cerebral)
    0,  # HeartDiseaseorAttack (sin enfermedad cardíaca)
    1,  # PhysActivity (actividad física realizada)
    1,  # Fruits (consume frutas diariamente)
    1,  # Veggies (consume vegetales diariamente)
    0,  # HvyAlcoholConsump (no consumo excesivo de alcohol)
    1,  # AnyHealthcare (tiene cobertura médica)
    0,  # NoDocbcCost (no tuvo problemas para pagar al médico)
    3,  # GenHlth (salud general: 3 = buena)
    2,  # MentHlth (días con problemas de salud mental en el último mes)
    0,  # PhysHlth (días con problemas de salud física en el último mes)
    0,  # DiffWalk (sin dificultad para caminar)
    1,  # Sex (1 = masculino)
    35, # Age (categoría de edad, por ejemplo, 35 = 35-39 años)
    4,  # Education (nivel educativo: 4 = graduado universitario)
    6   # Income (nivel de ingresos: 6 = $50,000-$74,999)
]])

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

print("Decrypted result is: ", result)