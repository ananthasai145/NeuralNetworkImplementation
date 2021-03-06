import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nn_utils import L_layer_model
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('classif_binary.csv').to_numpy()

# Extract features and target
X = dataset[:,0:8]
Y = dataset[:,8].reshape(-1,1)

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train).T
X_test = sc.transform(X_test).T
Y_train = Y_train.T
Y_test = Y_test.T

# Develop baseline model
layers_dims = [X_train.shape[0],4,Y_train.shape[0]]
base_params, base_costs = L_layer_model(X_train,Y_train,layers_dims,learning_rate=0.01,num_iterations=1000,print_cost=False)

# Plot Loss vs epochs
plt.figure()
plt.plot(base_costs)
plt.title('Loss vs epochs',fontweight='bold')
plt.xlabel('Epochs (X 100)')
plt.ylabel('Loss')
plt.show()
