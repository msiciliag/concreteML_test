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
        "http://127.0.0.1:5005/predict",
        files=files
    )

    return client.deserialize_decrypt_dequantize(encrypted_response.content)

#examples (ai generated)
X_low_risk =  np.array([[
    # --- Medias ---
    11.5,  # radius_mean
    16.0,  # texture_mean
    75.0,  # perimeter_mean
    400.0, # area_mean
    0.090, # smoothness_mean
    0.080, # compactness_mean
    0.040, # concavity_mean
    0.020, # concave points_mean
    0.170, # symmetry_mean
    0.060, # fractal_dimension_mean
    # --- Errores Estándar (SE) ---
    0.25,  # radius_se
    0.9,   # texture_se
    1.8,   # perimeter_se
    20.0,  # area_se
    0.006, # smoothness_se
    0.018, # compactness_se
    0.020, # concavity_se
    0.008, # concave points_se
    0.015, # symmetry_se
    0.0025,# fractal_dimension_se
    # --- "Peores" o Máximos ---
    13.0,  # radius_worst
    20.0,  # texture_worst
    85.0,  # perimeter_worst
    520.0, # area_worst
    0.120, # smoothness_worst
    0.180, # compactness_worst
    0.150, # concavity_worst
    0.070, # concave points_worst
    0.270, # symmetry_worst
    0.075  # fractal_dimension_worst
]], dtype=np.float64)
result_low_risk = make_request(client, X_low_risk)
print(f"Low risk malignant diagnosis example: ", result_low_risk)

X_high_risk = np.array([[
    # --- Medias ---
    20.0,  # radius_mean
    25.0,  # texture_mean
    130.0, # perimeter_mean
    1200.0,# area_mean
    0.110, # smoothness_mean
    0.180, # compactness_mean
    0.200, # concavity_mean
    0.100, # concave points_mean
    0.210, # symmetry_mean
    0.070, # fractal_dimension_mean
    # --- Errores Estándar (SE) ---
    0.70,  # radius_se
    1.5,   # texture_se
    5.0,   # perimeter_se
    80.0,  # area_se
    0.009, # smoothness_se
    0.040, # compactness_se
    0.050, # concavity_se
    0.018, # concave points_se
    0.025, # symmetry_se
    0.005, # fractal_dimension_se
    # --- "Peores" o Máximos ---
    25.0,  # radius_worst
    33.0,  # texture_worst
    160.0, # perimeter_worst
    1800.0,# area_worst
    0.150, # smoothness_worst
    0.400, # compactness_worst
    0.500, # concavity_worst
    0.200, # concave points_worst
    0.350, # symmetry_worst
    0.100  # fractal_dimension_worst
]], dtype=np.float64)
result_high_risk = make_request(client, X_high_risk)
print(f"High risk malignant diagnosis example: ", result_high_risk)
