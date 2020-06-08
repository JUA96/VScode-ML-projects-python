# Load base packages:
import pandas as pd
import numpy as np

# Load other key packages
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning packages
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance, plot_tree

# Load in the kickstarter finance data
df = pd.read_csv('data/Border_Crossing_Entry_Data.csv')
df.head()
df.tail()

for col in df.columns: 
    print(col) 

# Take a look at the data types
df.dtypes

# Take a look at the descriptive statistics 
df.describe()

# What is the mean size goal for US border crossings?
df['Value'].mean()

