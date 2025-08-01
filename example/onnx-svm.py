import numpy as np
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import pprint

print("starting")

X, y = make_classification(
    n_samples=100,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

print(X)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("train data")

clf = svm.SVC(kernel='rbf', gamma=0.5, C=1.0, probability=False)
clf.fit(X_train, y_train)

print("trained")

initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)

with open("svm_rbf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

#print("Support vectors:", clf.support_vectors_)
print("Support vectors (as Python list):")
pprint.pprint(clf.support_vectors_.tolist())
print("Dual coefficients shape:")
pprint.pprint(clf.dual_coef_.tolist())
print("Intercept:", clf.intercept_)
print("Gamma:", clf._gamma)
