#Import Required Modules
import pandas as pd
import pickle
import numpy as np

#Import Neural Network Module Functions
from custom_neural_net_creator.model import Model
from custom_neural_net_creator.dense import Dense
from custom_neural_net_creator.activation_layer import ActivationLayer
from custom_neural_net_creator.activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative
from custom_neural_net_creator.loss_functions import mean_squared_error, mean_squared_error_derivative

#Import configuration variables
from config import epoch_increment, max_epoch, trials

#Import variables from data processing
f = open('X_test.pckl', 'rb')
X_test = pickle.load(f)
f.close()

f = open('X_train.pckl', 'rb')
X_train = pickle.load(f)
f.close()

f = open('Y_test.pckl', 'rb')
Y_test = pickle.load(f)
f.close()

f = open('Y_train.pckl', 'rb')
Y_train = pickle.load(f)
f.close()

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

trial_data = [] #Store testing accuracies for each trial to save in csv

#Create model
model = Model()

model.add(Dense(15,32))
model.add(ActivationLayer(relu, relu_derivative))
model.add(Dense(32,32))
model.add(ActivationLayer(relu, relu_derivative))
model.add(Dense(32,1))
model.add(ActivationLayer(sigmoid, sigmoid_derivative))


model.fit(X_train, Y_train, mean_squared_error, mean_squared_error_derivative,100,learning_rate=0.1,verbosity=3)
predictions = model.predict(X_test)

total = len(predictions)
correct = 0
incorrect = 0

for i in range(total):
    prediction = round(predictions[i][0][0])
    if prediction == Y_test[i][0]:
        correct += 1
    else:
        incorrect += 1
accuracy_score = correct/total * 100
print("Percent Accuracy: " + str(accuracy_score))