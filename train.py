from concrete.ml.sklearn import LogisticRegression
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
import numpy as np

fhe_directory = '/tmp/fhe_client_server_files/'

model = LogisticRegression()

X = np.random.rand(100, 20)
y = np.random.randint(0, 2, size=100)

model.fit(X, y)
model.compile(X)

dev = FHEModelDev(path_dir=fhe_directory, model=model)
dev.save()