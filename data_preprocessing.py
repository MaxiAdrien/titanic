# The NameLength, TicketLength, AgeMissing and TicketFirstCharacter features were inspired by
# https://www.kaggle.com/code/zlatankr/titanic-random-forest-82-78

# The FamilySize feature was inspired by
# https://www.kaggle.com/code/vincentlugat/titanic-neural-networks-keras-81-8

# Imports

import pandas as pd
import numpy as np

# Load data

train = pd.read_csv('inputs/train.csv')
test = pd.read_csv('inputs/test.csv')

# Label train and test data

train['Label'] = 'train'
test['Label'] = 'test'

# Concatenate train and test data

df = pd.concat([train, test])

# Create features matrix

X = df.drop(['Survived', 'PassengerId'], axis=1)

# Feature engineering - Class

X['FirstClass'] = np.where(X['Pclass'] == 1, 1, 0)
X['SecondClass'] = np.where(X['Pclass'] == 2, 1, 0)

# Feature engineering - Title

X['Title'] = X['Name'].str.extract('([A-Za-z]+)\.', expand=False)

X['Title'] = X['Title'].replace({'Mme': 'Mrs',
                                 'Ms': 'Miss',
                                 'Mlle': 'Miss'})

X['Mr'] = np.where(X['Title'] == 'Mr', 1, 0)
X['Miss'] = np.where(X['Title'] == 'Miss', 1, 0)
X['Mrs'] = np.where(X['Title'] == 'Mrs', 1, 0)
X['Master'] = np.where(X['Title'] == 'Master', 1, 0)

# Feature engineering - Name length

X['NameLength'] = X['Name'].str.len()

# Feature engineering - Sex

X['Female'] = np.where(X['Sex'] == 'female', 1, 0)

# Feature engineering - Age data missing

X['AgeMissing'] = np.where(X['Age'].isna(), 1, 0)

# Feature engineering - Child

X['Child'] = np.where(X['Age'] < 18, 1, 0)

# Feature engineering - Family size

X['FamilySize'] = X['Parch'] + X['SibSp'] + 1

X['FamilySizeCat'] = np.where(X['FamilySize'] > 1, 'Small', 'Alone')
X['FamilySizeCat'] = np.where(X['FamilySize'] > 4, 'Big', X['FamilySizeCat'])

X['FamilySizeSmall'] = np.where(X['FamilySizeCat'] == 'Small', 1, 0)
X['FamilySizeBig'] = np.where(X['FamilySizeCat'] == 'Big', 1, 0)

# Feature engineering - Ticket number length

X['TicketLength'] = X['Ticket'].str.len()

# Feature engineering - First character of ticket number

X['TicketFirstCharacter'] = X['Ticket'].str[0]

X['TicketFirstCharacter3'] = np.where(X['TicketFirstCharacter'] == '3', 1, 0)
X['TicketFirstCharacter2'] = np.where(X['TicketFirstCharacter'] == '2', 1, 0)
X['TicketFirstCharacter1'] = np.where(X['TicketFirstCharacter'] == '1', 1, 0)
X['TicketFirstCharacterP'] = np.where(X['TicketFirstCharacter'] == 'P', 1, 0)
X['TicketFirstCharacterS'] = np.where(X['TicketFirstCharacter'] == 'S', 1, 0)
X['TicketFirstCharacterC'] = np.where(X['TicketFirstCharacter'] == 'C', 1, 0)
X['TicketFirstCharacterA'] = np.where(X['TicketFirstCharacter'] == 'A', 1, 0)
X['TicketFirstCharacterW'] = np.where(X['TicketFirstCharacter'] == 'W', 1, 0)
X['TicketFirstCharacter4'] = np.where(X['TicketFirstCharacter'] == '4', 1, 0)

# Feature engineering - Cabin letter

X['CabinCategory'] = X['Cabin'].apply(lambda x: str(x)[0])

X['CabinA'] = np.where(X['CabinCategory'] == 'A', 1, 0)
X['CabinB'] = np.where(X['CabinCategory'] == 'B', 1, 0)
X['CabinC'] = np.where(X['CabinCategory'] == 'C', 1, 0)
X['CabinD'] = np.where(X['CabinCategory'] == 'D', 1, 0)
X['CabinE'] = np.where(X['CabinCategory'] == 'E', 1, 0)
X['CabinF'] = np.where(X['CabinCategory'] == 'F', 1, 0)

X['CabinMissing'] = np.where(X['Cabin'].isna(), 1, 0)

# Feature engineering - Cabin number

X['CabinNumber'] = X['Cabin'].apply(lambda x: str(x).split(' ')[-1])
X['CabinNumber'] = np.where(X['CabinNumber'] == 'nan', np.nan, X['CabinNumber'].str[1:])
X['CabinNumber'] = pd.to_numeric(X['CabinNumber'])
X['CabinNumberLow'] = np.where(X['CabinNumber'] <= 30, 1, 0)
X['CabinNumberMed'] = np.where((X['CabinNumber'] > 30) & (X['CabinNumber'] <= 63), 1, 0)
X['CabinNumberHigh'] = np.where(X['CabinNumber'] > 63, 1, 0)

# Replace missing values for Embarked with mode

X['Embarked'] = X['Embarked'].fillna('S')

# Feature engineering - Embarked

X['EmbarkedC'] = np.where(X['Embarked'] == 'C', 1, 0)
X['EmbarkedQ'] = np.where(X['Embarked'] == 'Q', 1, 0)
X['EmbarkedS'] = np.where(X['Embarked'] == 'S', 1, 0)

# Replace missing age with training set median (by title and class)

X['Age'] = X[X['Label'] == 'train'].groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# Replace missing fare with training set median (by class)

median_fare = X['Fare'][(X['Label'] == 'train') & (X['Pclass'] == 3)].median()
X['Fare'] = X['Fare'].fillna(median_fare)

# Split data into train and test

X_train = X[X['Label'] == 'train']
X_test = X[X['Label'] == 'test']

# Save features matrix (prior to dropping unused features) to csv

X_train.to_csv('intermediary_outputs/X_train_all.csv')

# Drop unused features

X_train = X_train.drop(['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Label', 'Title',
                        'FamilySize', 'FamilySizeCat', 'TicketFirstCharacter', 'CabinCategory', 'CabinNumber'], axis=1)
X_test = X_test.drop(['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Label', 'Title',
                      'FamilySize', 'FamilySizeCat', 'TicketFirstCharacter', 'CabinCategory', 'CabinNumber'], axis=1)

# Count and print features used

features_used = X_train.columns.tolist()

print('Number of features used:', len(features_used))
print('Features used:', features_used)

# Save features matrix (after dropping unused features) to csv

X_train.to_csv('intermediary_outputs/X_train.csv', index=False)
X_test.to_csv('intermediary_outputs/X_test.csv', index=False)
