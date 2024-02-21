#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:08:59 2024

@author: raghad
"""
#slides: https://www.canva.com/design/DAF9ZfUHBaw/WT3HCFIyky2DJmP9u4CnbA/edit?utm_content=DAF9ZfUHBaw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
#DataSets:
#https://www.kaggle.com/datasets/dermisfit/fraud-transactions-dataset/data

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from qiskit import Aer, transpile, assemble
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel

# Load the dataset
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

# Concatenate train and test data for preprocessing
data = pd.concat([train_data, test_data], ignore_index=True)

# Handle missing values
data.fillna(0, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

# Scale numerical features
scaler = StandardScaler()
data[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']] = scaler.fit_transform(data[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']])

# Split the data into training and test sets
X_train = data.iloc[:len(train_data)].drop('is_fraud', axis=1)
y_train = data.iloc[:len(train_data)]['is_fraud']
X_test = data.iloc[len(train_data):].drop('is_fraud', axis=1)
y_test = data.iloc[len(train_data):]['is_fraud']

# Classical Feature Selection
feature_selector = SelectKBest(chi2, k=5)  # Select top 5 features based on chi-squared test
X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_test_selected = feature_selector.transform(X_test)

# Quantum Feature Mapping
seed = 12345
backend = Aer.get_backend('statevector_simulator')

# Define the quantum feature map
feature_map = ZZFeatureMap(feature_dimension=len(X_train_selected.columns), reps=2, entanglement='linear')

# Define the quantum kernel
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend, seed_simulator=seed, seed_transpiler=seed)

# Define the variational quantum classifier
vqc = VQC(quantum_kernel, RandomForestClassifier(n_estimators=100, random_state=42))

# Train the Quantum Random Forest Classifier
vqc.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = vqc.predict(X_test_selected)

# Evaluate the model on the test set
print(classification_report(y_test, y_pred))
