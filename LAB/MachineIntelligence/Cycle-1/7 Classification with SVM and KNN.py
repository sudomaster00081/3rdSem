# Download any OCR dataset and perform the classification with SVM and KNN. Compare the obtained result


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors, metrics
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist.data.astype('int'))
y = np.array(mnist.target.astype('int'))
# Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SVM
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
# KNN
knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
# Comparison
svm_accuracy = metrics.accuracy_score(y_test, svm_predictions)
knn_accuracy = metrics.accuracy_score(y_test, knn_predictions)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"KNN Accuracy: {knn_accuracy:.4f}")
# Display
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'SVM: {svm_predictions[i]}\nKNN: {knn_predictions[i]}')
    ax.axis('off')

plt.show()
