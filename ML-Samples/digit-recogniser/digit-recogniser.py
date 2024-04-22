# Following implementation is based on the book Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow 3rd Edition by Aurélien Géron

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

print(type(X[0]))
print(y)

import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

test_digit = X[0]
plot_digit(test_digit)
plt.show()

# MNIST dataset is already split into training set (first 60000) and test set (last 10000), with good shuffled test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# let us first do a binary classification like distinguish 5 and non-5

# set True to all 5s and False to others
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)

print(sgd_classifier.predict([test_digit]))

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_classifier, X_train, y_train, cv=3, scoring='accuracy'))

from sklearn.dummy import DummyClassifier
dummy_classifier = DummyClassifier()
dummy_classifier.fit(X_train, y_train_5)
print(any(dummy_classifier.predict(X_train)))
print(cross_val_score(dummy_classifier, X_train, y_train, cv=3, scoring='accuracy'))