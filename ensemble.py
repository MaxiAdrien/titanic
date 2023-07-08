# Imports

import pandas as pd
import numpy as np

# Load individual predictions

nn = pd.read_csv('outputs/submission_nn.csv')
rf = pd.read_csv('outputs/submission_rf.csv')
svm = pd.read_csv('outputs/submission_svm.csv')

# Calculate and print percentage of identical predictions for each pair of models

pct_identical_nn_rf = np.mean(nn['Survived'] == rf['Survived'])
pct_identical_nn_svm = np.mean(nn['Survived'] == svm['Survived'])
pct_identical_rf_svm = np.mean(rf['Survived'] == svm['Survived'])

print('Percentage of identical predictions between NN and RF:', pct_identical_nn_rf.round(2))
print('Percentage of identical predictions between NN and SVM:', pct_identical_nn_svm.round(2))
print('Percentage of identical predictions between RF and SVM:', pct_identical_rf_svm.round(2))

# Ensemble predictions

submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['Survived'] = pd.concat([nn['Survived'], rf['Survived'], svm['Survived']], axis=1).mode(axis=1)
submission['PassengerId'] = submission.index + 892

# Export to csv

submission.to_csv('outputs/submission_ensemble.csv', index=False)
