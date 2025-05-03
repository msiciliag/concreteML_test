"""
This script is used to send encrypted data to the REST API, receive the encrypted response, and decrypt it.
"""

from concrete.ml.deployment import FHEModelClient
import numpy as np
import requests

FHE_DIRECTORY = '/tmp/fhe_client_server_files/'

client = FHEModelClient(path_dir=FHE_DIRECTORY, key_dir="/tmp/keys_client")

def make_request(client, X_new):
    encrypted_data = client.quantize_encrypt_serialize(X_new)
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()

    files = {
        'encrypted_data': ('encrypted_data.bin', encrypted_data, 'application/octet-stream'),
        'evaluation_keys': ('evaluation_keys.bin', serialized_evaluation_keys, 'application/octet-stream')
    }

    encrypted_response = requests.post(
        "http://127.0.0.1:5001/predict",
        files=files
    )

    return client.deserialize_decrypt_dequantize(encrypted_response.content)

#examples (ai generated)
X_low_risk = np.array([[
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
result_low_risk = make_request(client, X_low_risk)
print(result_low_risk)

X_normal_risk = np.array([[
    1,  # HighBP
    1,  # HighChol
    1,  # CholCheck
    40, # BMI (higher BMI for higher risk)
    1,  # Smoker
    1,  # Stroke
    1,  # HeartDiseaseorAttack
    0,  # PhysActivity (no physical activity)
    0,  # Fruits (does not consume fruits)
    0,  # Veggies (does not consume vegetables)
    1,  # HvyAlcoholConsump (heavy alcohol consumption)
    0,  # AnyHealthcare (no healthcare coverage)
    1,  # NoDocbcCost (could not afford doctor)
    5,  # GenHlth (poor general health)
    15, # MentHlth (many days with mental health issues)
    20, # PhysHlth (many days with physical health issues)
    1,  # DiffWalk (difficulty walking)
    0,  # Sex (0 = female)
    65, # Age (older age category, e.g., 65-69 years)
    1,  # Education (lower education level)
    1   # Income (lower income level)
]])
result_high_risk = make_request(client, X_normal_risk)
print(result_high_risk)