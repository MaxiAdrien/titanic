# Inspired by https://www.kaggle.com/code/zlatankr/titanic-random-forest-82-78

# Imports

import pandas as pd
import numpy as np

# Load data

train = pd.read_csv('inputs/train.csv')
test = pd.read_csv('inputs/test.csv')

# Summary of training data

print(train.info(memory_usage=False))

# Glance at data

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

print(train.head(10))

# Percentage of people who died

print(train['Survived'].value_counts(normalize=True).round(2))

# Survival per class

print(train['Survived'].groupby(train['Pclass']).mean().round(2))

# Distribution of titles

train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)

print(train['Survived'].groupby(train['Title']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Survival by name length

train['NameLength'] = train['Name'].str.len()
print(train['Survived'].groupby(pd.qcut(train['NameLength'], 5)).mean().round(2))

# Survival by sex

print(train['Survived'].groupby(train['Sex']).mean().round(2))

# Relationship between survival and whether age data is missing

print(train['Survived'].groupby(train['Age'].notna()).mean().round(2))

# Survival by age

print(train['Survived'].groupby(pd.qcut(train['Age'], 5)).mean().round(2))

# Survival by number of siblings on board

print(train['Survived'].groupby(train['SibSp']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Survival by number of parents/children on board

print(train['Survived'].groupby(train['Parch']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Survival by family size

train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
print(train['Survived'].groupby(train['FamilySize']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Survival by length of ticket number

train['TicketLength'] = train['Ticket'].str.len()
print(train['Survived'].groupby(train['TicketLength']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Survival by first character of ticket number

train['TicketFirstCharacter'] = train['Ticket'].str[0]
print(train['Survived'].groupby(train['TicketFirstCharacter']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Survival by fare

print(train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean().round(2))

# Relationship between survival and whether cabin data is missing

print(train['Survived'].groupby(train['Cabin'].notna()).mean().round(2))

# Survival by cabin letter

train['CabinCategory'] = train['Cabin'].apply(lambda x: str(x)[0])
print(train['Survived'].groupby(train['CabinCategory']).mean().round(2))

# Survival by cabin number

train['CabinNumber'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1])
train['CabinNumber'] = np.where(train['CabinNumber'] == 'nan', np.nan, train['CabinNumber'].str[1:])
train['CabinNumber'] = pd.to_numeric(train['CabinNumber'])
print(train['Survived'].groupby(pd.qcut(train['CabinNumber'], 3)).mean().round(2))

# Survival by embarkment location

print(train['Survived'].groupby(train['Embarked']).mean().round(2))

# Age by class

print(train['Age'].groupby(train['Pclass']).mean().round(2))

# Age by sex

print(train['Age'].groupby(train['Sex']).mean().round(2))

# Age by title

print(train['Age'].groupby(train['Title']).agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Age by title and class

print(train.groupby(['Title', 'Pclass'])['Age'].agg(['count', 'mean']).round(2).reset_index().sort_values('count', ascending=False))

# Fare by class

print(train['Fare'].groupby(train['Pclass']).mean().round(2))

# Embarked value counts

print(train['Embarked'].value_counts())
