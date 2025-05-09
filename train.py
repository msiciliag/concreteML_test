'''
This script trains a Random Forest Classifier model and saves it 
following the ConcreteML deployment schema.
'''

import logging
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import RandomForestClassifier
from concrete.ml.deployment import FHEModelDev
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

fhe_directory = '/tmp/fhe_client_server_files/'
logger.info(f"FHE files will be saved to: {fhe_directory}")

model = RandomForestClassifier(n_estimators=10)
logger.info("Initialized Random Forest model")

logger.info("Fetching Diagnostic Wisconsin Breast Cancer dataset...")
ds = fetch_ucirepo(id=17)

X = ds.data.features
X = X.astype(float)
y = ds.data.targets
le = LabelEncoder()
y = le.fit_transform(y.values.ravel())
y = y.astype(int)

X = X.astype('float64')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#logger.info("Feature data types:")
#logger.info(X.dtypes)

#logger.info(f"Dataset shape: {X.shape} samples with {X.shape[1]} features")
logger.info(f"Training set size: {X_train.shape[0]} samples")
logger.info(f"Test set size: {X_test.shape[0]} samples")

#logger.info("Variables description:")
#logger.info(ds.variables)

logger.info("-----------------------------------")
logger.info("Training model...")
model.fit(X_train, y_train)
logger.info("-----------------------------------")

logger.info("Predicting with clear model...")
results = {}
tic = time.perf_counter()
y_pred_clear = model.predict(X_test)
toc = time.perf_counter()
results["prediction_time_clear"] = toc - tic
results["accuracy_clear"] = accuracy_score(y_test, y_pred_clear)
#results["confusion_matrix"] = confusion_matrix(y_test, y_pred_clear)
#results["classif_report"] = classification_report(y_test, y_pred_clear)

logger.info("-----------------------------------")
logger.info("Clear model evaluation:")
logger.info(f"Test set size: {X_test.shape[0]} samples")
logger.info(f"Accuracy: {results['accuracy_clear']:.4f}")
logger.info(f"Prediction time: {results['prediction_time_clear']:.4f} seconds")
logger.info("-----------------------------------")

#logger.info(f"Confusion matrix:\n{results['confusion_matrix']}")
#logger.info(f"Classification report:\n{results['classif_report']}")

logger.info("Compiling model for FHE...")
model.compile(X)
logger.info("-----------------------------------")

logger.info("Predicting with FHE model...")
tic = time.perf_counter()
y_pred_fhe = model.predict(X_test, fhe="execute")
toc = time.perf_counter()
results["prediction_time_fhe"] = toc - tic
results["accuracy_fhe"] = accuracy_score(y_test, y_pred_fhe)
#results["confusion_matrix_fhe"] = confusion_matrix(y_test, y_pred_fhe)
#results["classif_report_fhe"] = classification_report(y_test, y_pred_fhe)
logger.info("-----------------------------------")
logger.info("FHE model evaluation:")
logger.info(f"Test set size: {X_test.shape[0]} samples")
logger.info(f"Accuracy: {results['accuracy_fhe']:.4f}")
logger.info(f"Prediction time: {results['prediction_time_fhe']:.4f} seconds")
#logger.info(f"Confusion matrix:\n{results['confusion_matrix_fhe']}")
#logger.info(f"Classification report:\n{results['classif_report_fhe']}")
logger.info("-----------------------------------")


logger.info("Saving FHE model...")
#logger.info("Model classes: %s", model.classes_)
dev = FHEModelDev(path_dir=fhe_directory, model=model)
dev.save()
logger.info("FHE model saved successfully!")