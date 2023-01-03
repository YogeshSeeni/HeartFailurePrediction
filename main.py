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
from config import epoch_increment, max_epoch, trials, learning_rate, verbosity, hidden_neurons

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

#Create accuracy metric
def testing_accuracy(predictions):
    global Y_test

    total = len(predictions) #Get total amount of testcases in testing dataset
    correct = 0 #Track how many many testcases the model accurately predicts

    for i in range(total):
        prediction = round(predictions[i][0][0]) #Round the prediction of the model to 0 or 1
        if prediction == Y_test[i][0]: #Check to see if the prediction is equal to the actual data
            correct += 1

    accuracy_score = correct/total * 100 #Create a percent accuracy variable
    return accuracy_score

#Loop over the number of trials using range function
for trial in range(trials):
    print("Running Trial " + str(trial + 1)) #Output current trial being tested
    accuracies = [] #Keeps track of the accuracies for each epoch being recorded
    current_epoch = 0 #Keeps track of current epoch number being tested

    #Create model
    model = Model()

    model.add(Dense(15,hidden_neurons)) #15 neurons in input layer to a variable containing the number of hidden neurons in the hidden layer
    model.add(ActivationLayer(relu, relu_derivative)) #Use rectified linear unit as the activation function on the hidden layer
    model.add(Dense(hidden_neurons,hidden_neurons)) #Add an output layer containing 1 neuron
    model.add(ActivationLayer(relu, relu_derivative))
    model.add(Dense(hidden_neurons,1))
    model.add(ActivationLayer(sigmoid, sigmoid_derivative)) #Use sigmoid as activation function to scale output between 0 or 1

    while current_epoch <= max_epoch: #Make sure that model is only being trained up to the max_epoch
        #Train model on training data epoch_increment more times
        model.fit(X_train, Y_train, mean_squared_error, mean_squared_error_derivative,epoch_increment,learning_rate=learning_rate,verbosity=verbosity)

        #Get predictions on testing data and get accuracy
        predictions = model.predict(X_test)
        accuracy = testing_accuracy(predictions)

        accuracies.append(accuracy) #Add accuracy to end of the accuracies array
        current_epoch += epoch_increment #Add the epoch increment to the current epoch to be tested in next iteration

    trial_data.append(accuracies) #Add the data for this current trial to the overall data

#Output data in formatted csv

#Create Columns
columns = list(range(0,max_epoch+1,epoch_increment))
columns = [str(i) + " epoch" if i == 1 else str(i) + " epochs" for i in columns]

#Create Indexes
indexes = [f"Trial #{i+1}" for i in range(trials)] 

#Create Pandas Dataframe
df = pd.DataFrame(trial_data, columns=columns, index=indexes)

#Export to CSV
df.to_csv("./trial_data.csv")