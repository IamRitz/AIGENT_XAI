from matplotlib import pyplot
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(y_train.shape)

X_train = X_train.reshape((X_train.shape[0], 32*32*3)).astype('float32')
# X_test.shape[0], X_train[1]*X_train[2]*X_train[3]
X_test = X_test.reshape((X_test.shape[0], 32*32*3)).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

print(X_train.shape)
print(y_train.shape)

import csv
f = open('inputs.csv', 'w')
writer = csv.writer(f)

for i in range(X_train.shape[0]):
  # print(X_train[i])
  writer.writerow(X_train[i])
f.close()

f = open('outputs.csv', 'w')
writer = csv.writer(f)

for i in range(y_train.shape[0]):
  # print(X_train[i])
  writer.writerow(y_train[i])
f.close()