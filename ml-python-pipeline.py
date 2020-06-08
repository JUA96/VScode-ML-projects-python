#Load necessary packages:
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning packages
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

# Load in the kickstarter finance data
df = pd.read_csv('data/ks-projects.csv')
df.head()
df.tail()

for col in df.columns: 
    print(col) 

# Take a look at the data types
df.dtypes

# Take a look at the descriptive statistics 
df.describe()

# What is the mean size goal for kickstarter projects?
df['goal'].mean()

df['state'].value_counts(dropna=False)


