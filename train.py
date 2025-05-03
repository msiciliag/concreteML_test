'''
This script trains a Random Forest Classifier model and saves it following the ConcreteML deployment schema.
'''

from concrete.ml.sklearn import RandomForestClassifier
from concrete.ml.deployment import FHEModelDev
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

import time
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

print("Starting FHE Random Forest training process...")

fhe_directory = '/tmp/fhe_client_server_files/'
print(f"FHE files will be saved to: {fhe_directory}")

model = RandomForestClassifier(n_estimators=10) 
print("Initialized Random Forest model")

print("\nFetching CDC Diabetes Health Indicators dataset...")
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets
X = X.astype('float64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=42)

print("\nFeature data types:")
print(X.dtypes)

print(f"\nDataset shape: {X.shape} samples with {X.shape[1]} features")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Target distribution:\n{y.value_counts()}")
  
print("\nVariables description:")
print(cdc_diabetes_health_indicators.variables) 

print("\nTraining model...")
model.fit(X_train, y_train)

print("\nPredicting with clear model...")
results = {}
tic = time.perf_counter()
y_pred_clear = model.predict(X_test)
toc = time.perf_counter()
results["prediction_time_clear"] = toc - tic
results["accuracy_clear"] = accuracy_score(y_test, y_pred_clear)
results["f1_clear"] = f1_score(y_test, y_pred_clear)
results["auc_clear"] = roc_auc_score(y_test, y_pred_clear)
print("\nClear model evaluation:")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Accuracy: {results['accuracy_clear']:.4f}")
print(f"F1 Score: {results['f1_clear']:.4f}")
print(f"AUC: {results['auc_clear']:.4f}")
print(f"Prediction time: {results['prediction_time_clear']:.4f} seconds")

print("\nCompiling model for FHE...")
model.compile(X)

print("\nPredicting with FHE model...")
tic = time.perf_counter()
y_pred_fhe = model.predict(X_test, fhe="execute")
toc = time.perf_counter()
results["prediction_time_fhe"] = toc - tic
results["accuracy_fhe"] = accuracy_score(y_test, y_pred_fhe)
results["f1_fhe"] = f1_score(y_test, y_pred_fhe)
results["auc_fhe"] = roc_auc_score(y_test, y_pred_fhe)
print("\nFHE model evaluation:")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Accuracy: {results['accuracy_fhe']:.4f}")
print(f"F1 Score: {results['f1_fhe']:.4f}")
print(f"AUC: {results['auc_fhe']:.4f}")
print(f"Prediction time: {results['prediction_time_fhe']:.4f} seconds")

print("\nSaving FHE model...")
print("Model classes:", model.classes_)
dev = FHEModelDev(path_dir=fhe_directory, model=model)
dev.save()
print("FHE model saved successfully!")