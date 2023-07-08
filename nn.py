# Imports

import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import time

# Time script execution

start = time.time()

# Load data

train = pd.read_csv('inputs/train.csv')
X_train = pd.read_csv('intermediary_outputs/X_train.csv')
X_test = pd.read_csv('intermediary_outputs/X_test.csv')

# Target

y = to_categorical(train['Survived'])

# Neural network model

n_cols = X_train.shape[1]

model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(n_cols,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y, validation_split=0.30, batch_size=32, epochs=20)

# Make predictions

probabilities = model.predict(X_test)
threshold = 0.5

submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['Survived'] = np.where(probabilities[:, 1] > threshold, 1, 0)
submission['PassengerId'] = submission.index + 892

# Export to csv

submission.to_csv('outputs/submission_nn.csv', index=False)

# Print script execution time

print('Running script took', round(time.time()-start, 1), 'seconds.')
