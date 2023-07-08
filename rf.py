# Imports

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Random forests model

rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=188)

# Hyperparameter tuning

tuned_parameters = [{'n_estimators': [300, 500, 700, 900],
                     'criterion': ['gini', 'entropy'],
                     'min_samples_split': [5, 6, 7, 8, 9],
                     'min_samples_leaf': [1, 2, 3, 4],
                     'max_features': [0.20, 0.225, 0.25, 0.275, 0.30]}]

model = GridSearchCV(rf, tuned_parameters, cv=3, verbose=3)
model.fit(X_train, y)

scores = model.cv_results_['mean_test_score']
scores_std = model.cv_results_['std_test_score']

print('Average scores:', scores.round(4))
print('Score standard deviations:', scores_std.round(3))
print('Best parameters:', model.best_params_)
print('Best score:', round(model.best_score_, 4))

best_rf = RandomForestClassifier(criterion='entropy',
                                 max_features=0.225,
                                 min_samples_leaf=1,
                                 min_samples_split=7,
                                 n_estimators=700,
                                 oob_score=True,
                                 n_jobs=-1)

best_rf.fit(X_train, y)

# Feature importances

feature_importances = pd.Series(best_rf.feature_importances_.round(2), index=X_train.columns)

print('FEATURE IMPORTANCES (TOP 15):')
print(feature_importances.sort_values(ascending=False).head(15).to_string())

# Make predictions

submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['Survived'] = model.predict(X_test)
submission['PassengerId'] = submission.index + 892

# Export to csv

submission.to_csv('outputs/submission_rf.csv', index=False)

# Print script execution time

print('Running script took', round(time.time()-start, 1), 'seconds.')
