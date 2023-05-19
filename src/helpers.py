
#https://github.com/interpretml/DiCE/blob/main/dice_ml/utils/helpers.py
import os
import pickle
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def load_adult_income_dataset(only_train=True):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                             delimiter=', ', dtype=str, invalid_raise=False)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'relationship','income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if only_train:
        train, _ = train_test_split(adult_data, test_size=0.2, random_state=17)
        adult_data = train.reset_index(drop=True)

    # Remove the downloaded dataset
    if os.path.isdir('archive.ics.uci.edu'):
        entire_path = os.path.abspath('archive.ics.uci.edu')
        shutil.rmtree(entire_path)
    adult_data=adult_data[['gender','marital_status',  'relationship','occupation','income' ]]
    adult_data.columns=['S','M','R','O','I']
    return adult_data

def load_compass_data():
    # Load the dataset
    data = pd.read_csv('../data/compas-scores-two-years.csv')

    # Choose the columns you want to use for training
    # These might not be the right columns for your specific use case
    feature_columns = ['age', 'c_charge_degree', 'race', 'sex', 'priors_count','two_year_recid']

    # Prepare the features
    df = data[feature_columns]
    df=df[["race","sex","priors_count","two_year_recid"]]
    df.columns=["A","C","M","Y"]
    return df

def label_enocde(df):
    original_df=df.copy()
    for col in df.columns[df.dtypes == 'object']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(le.classes_)
    return original_df,df

def train_model(df,kind=None):
    if kind=="compass":
        y = df["Y"]
        X= df[["A","M","C"]]
    elif kind=="adult":
        y=df['I']
        X=df[['S','M','R','O']]
# Train a logistic regression classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Test the classifier
    y_pred = clf.predict(X)
    print('Accuracy:', accuracy_score(y, y_pred))
    return clf