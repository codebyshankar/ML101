# Following implementation is based on the book Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow 3rd Edition
# by Aurélien Géron

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target

print(X)
print(y)

import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

test_digit = X[0]
plot_digit(test_digit)
plt.show()