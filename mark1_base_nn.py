import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from nn_utils import L_layer_model
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('classif_binary.csv').to_numpy()
#print("Size of the dataset is:",dataset.shape)
X = dataset[:,0:8]
Y = dataset[:,8].reshape(-1,1)
#print("Size of features:",X.shape)
#print("Size of labels:",Y.shape)
# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
#print("Size of train features:",X_train.shape)
#print("Size of train labels:",Y_train.shape)
#print("Size of test features:",X_test.shape)
#print("Size of test labels:",Y_test.shape)
# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train).T
X_test = sc.transform(X_test).T
Y_train = Y_train.T
Y_test = Y_test.T
print("Size of train features:",X_train.shape)
print("Size of train labels:",Y_train.shape)
print("Size of test features:",X_test.shape)
print("Size of test labels:",Y_test.shape)
# Develop baseline model
layers_dims = [X_train.shape[0],8,8,8,8,Y_train.shape[0]]
#layers_dims = [X_train.shape[0],4,Y_train.shape[0]]
base_params, base_costs = L_layer_model(X_train,Y_train,layers_dims,learning_rate=0.01,num_iterations=1000,print_cost=False)
# Plot Loss vs epochs
plt.figure()
plt.plot(base_costs)
plt.title('Loss vs epochs',fontweight='bold')
plt.xlabel('Epochs (X 100)')
plt.ylabel('Loss')
plt.show()