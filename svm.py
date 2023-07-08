# Imports

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# Time script execution

start = time.time()

# Load data

train = pd.read_csv('inputs/train.csv')
X_train = pd.read_csv('intermediary_outputs/X_train.csv')
X_test = pd.read_csv('intermediary_outputs/X_test.csv')

# Target

y = train['Survived']

# Pipeline - feature scaling and SVC model

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('SVM', SVC())])

# Hyperparameter tuning

tuned_parameters = [{'SVM__C': [0.1, 1, 10],
                     'SVM__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'SVM__degree': [1, 2, 3, 4],
                     'SVM__gamma': ['scale', 'auto']}]
model = GridSearchCV(pipeline, tuned_parameters, cv=5, verbose=1)
model.fit(X_train, y)

scores = model.cv_results_['mean_test_score']
scores_std = model.cv_results_['std_test_score']
print('Average scores:', scores.round(4))
print('Score standard deviations:', scores_std.round(4))
print('Best parameters:', model.best_params_)
print('Best score:', round(model.best_score_, 4))

# Make predictions

submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['Survived'] = model.predict(X_test)
submission['PassengerId'] = submission.index + 892

# Export to csv

submission.to_csv('outputs/submission_svm.csv', index=False)

# Print script execution time

print('Running script took', round(time.time()-start, 1), 'seconds.')
