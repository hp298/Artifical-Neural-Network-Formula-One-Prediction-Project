'''
CS3220 F1 Data Project

Hermogenes Parente hp298@cornell.edu

Data Sources:
Formula 1 Histroic Data: http://Ergast.com/mrd
Historic Weather Data: https://openweathermap.org/history

Code Sources:
https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
https://pytorch.org/docs/stable/nn.html

Input Data Set: List[List[]] with n rows and row[n-1] (the last one) \
    being the output row 
'''
import os
import time
from datetime import datetime
import numpy as np
from numpy import linalg as la
import json
from sklearn.decomposition import PCA
import torch

class SetupDataset(torch.utils.data.dataset.Dataset):
    # load in data set from txt
    def __init__(self, file_name, **kwargs):
        # read the file
        with open(file_name, "r") as dat:
            print("Retrieving Data Set from Local File...")
            data_set= json.load(dat)

        # usefule variables for setting up tensors
        self.size= len(data_set)
        self.in_size= len(data_set[0])-1
        self.out_size= 1

        # set pca dim if in argument
        new_dim= min(abs(int(kwargs.get('dim',self.in_size))),self.in_size)

        # make tensors for samples and actual results
        self.input_data= torch.empty((self.size,self.in_size),dtype=torch.float32)
        self.output_data= torch.empty((self.size,self.out_size),dtype=torch.float32)

        for idx in range(self.size):
            # add sample to input set
            temp_tensor= torch.FloatTensor(data_set[idx][:self.in_size])
            self.input_data[idx]= temp_tensor

            # find the actual result (last row in the vector)
            pos= data_set[idx][self.in_size]
            # add to output set
            self.output_data[idx][0]= float(pos)

        # conduct PCA on input
        self.input_data= calcPCA(self.input_data,new_dim)
        self.in_size= new_dim

    # rows in the dataset
    def __len__(self):
        return len(self.input_data)
    
    # gets the input/output data at the index
    def __getitem__(self, index):
        return [self.input_data[index], self.output_data[index]]

    # set size
    def getSize(self):
        return self.size

    # input size
    def getInputSize(self):
        return self.in_size

    # outside size
    def getOutputSize(self):
        return self.out_size

class MLModel(torch.nn.Module):

    # the ANN model used to define 
    def __init__(self, input_size, output_size):
        '''
        Documention: https://pytorch.org/docs
        The Model:
        Input    Hidden     Output
        O----\ 
              X
        O----/ X----O----\ 
        ...        ...    X---O
        O----\ X----O----/
              X
        O----/
        '''
        super(MLModel, self).__init__()

        # number of neurons in the hidden layer
        self.hidden_size= int((input_size+1) * 2 / 3)

        # linear layers used in the model
        self.layer_hidden= torch.nn.Linear(input_size, self.hidden_size)
        self.layer_out= torch.nn.Linear(self.hidden_size, output_size)

        # activation function used on the neurons
        self.activation= torch.nn.Sigmoid()
    
    # propagate the sample forward
    def forward(self, x):
        # move input to the hidden layer
        x= self.layer_hidden(x)

        # active the output of the output layer
        x= self.activation(x)

        # move activated output to output layer
        x= self.layer_out(x)

        # return the predicted output
        return x
    
class ProgressBar:
    # initialize progress bar when called
    def __init__(self, prefix, total):
        self.prefix= prefix
        self.barLength= 50
        self.total= total
        self.cur= 0
        self.percent= 0
        self.arrow= '█' * int(0) + '█'
        self.spaces= ' ' * (self.barLength - len(self.arrow))
    
    # updates and iterates bar when called
    def __call__(self):
        self.cur+=1
        self.percent= float(self.cur) * 100 / self.total
        self.arrow= '█' * int(self.percent/100 * self.barLength - 1) + '█'
        self.spaces= self.spaces  = ' ' * (self.barLength - len(self.arrow))
        self.draw()

    # draws the updated progress bar
    def draw(self):
        print(self.prefix + ': |%s%s| %d %%' % (self.arrow, self.spaces, self.percent), end='\r')

    # skip line
    def end(self):
        print('')

class Timer:
    # starts timer when created
    def __init__(self):
        self.start_time= time.process_time()

    # calculates elapsed time from when timer created to this call
    def __call__(self):
        self.end_time= time.process_time()
        self.time_elapsed= self.end_time - self.start_time
        self.printTime()

    # prints the elapsed time
    def printTime(self):
        print("Time Elapsed (seconds): %f"%self.time_elapsed)

def calcPCA(input_set, new_dim):
    # uses sklearn PCA function to computer PCA of the input set
    
    # Turns PyTorch Tensor into numpy array
    np_arr= input_set.numpy()

    # creates PCA for desired num of dimensions
    pca= PCA(new_dim)

    # fits input data and transforms it
    new_input= pca.fit_transform(np_arr)

    # convert back to tensor for the model
    tensor_input= torch.from_numpy(new_input)

    # return the tensor
    return tensor_input

def meanSquareError(observed_values, predicted_values):
    # calc mean square error to evelauate predicitons of model
    try:
        # first, need predicted and observed results to be same dim
        observed_size= len(observed_values)
        predicted_size= len(predicted_values)
        assert observed_size == predicted_size
        data_points= observed_size
        mse_sum= 0

        # sum square diff of observed and predicted fininshing position
        for i in range(data_points):
            mse_sum+= (observed_values[i] - predicted_values[i][0]) ** 2
        
        calculated_mse= mse_sum / data_points

        # prints the importnat values calculated
        print("\nData Points: %i" % data_points)
        print("MSE Sum Total: %f" % mse_sum)
        print("MSE Calculated: %f"% calculated_mse)

        # return average for all data points
        return calculated_mse
    
    # when dimesions are different
    except AssertionError as error:
        print("Inputs must be the same size; %i != %i " % (observed_size, predicted_size))
        return None

def trainModel(model, training_set, training_set_size, epoch, batch):
    # starts the timer for evaluating time to train
    timer= Timer()

    # loss function
    loss= torch.nn.MSELoss()

    # optimization function
    optimization= torch.optim.Adam(model.parameters())

    # creates the progress bar - indicating % of training completed
    progress_bar= ProgressBar("Sets Trained",training_set_size * epoch)

    # epochs - number of times to go through training data set
    for e in range(epoch):

        # batch - size of each enumerate
        for i, (samples, observed) in enumerate(training_set):
            # clear the gradient for a new batch
            optimization.zero_grad()

            # predited output
            predicted= model(samples)

            # finds diff of predicted and expected outputs
            difference= loss(predicted, observed)
            
            # backpropogate error through the model
            difference.backward()

            # update the model weights
            optimization.step()

            # update prorgess bar
            progress_bar()

    # stop progress bar and print elapsed time
    progress_bar.end()
    timer()

def testModel(model, testing_set, testing_set_size, output_size):
    # starts the timer for evaluating time to test
    timer= Timer()
    
    # variables for evaluating model
    predicted_set= torch.empty((testing_set_size,output_size),dtype= torch.float32)
    observed_set= torch.empty((testing_set_size,output_size),dtype= torch.float32)

    # test the model
    for i, (samples, observed) in enumerate(testing_set):
        
        # make two sets
        predicted= model(samples)
        predicted_set[i]= predicted
        observed_set[i]= observed

    # evaluate the model
    meanSquareError(observed_set, predicted_set)

    # prints elapsed time
    timer()

def makePrediciton(sample, print=True):
    '''
    Runs the given sample through the model
    Prints the result if true (optional)
    '''
    predicted= model(sample)

    if print:
        print("\nPredicted Race Result: %i" % predicted)
    
    return predicted
    
'''
Machine Learning - Neural Networks using PyTorch
A way to predict the results of an Formula 1 Race
Using race data - date, track, weather, drivers, teams, and thier stats
'''
print("\n______________________")
print("Starting ML Program...\n")

# const vars
DATA_FILE_PATH= 'Final Deliverable\Formula-Data-Project\TrainingData.txt'
DIM_REDUCTION= 12
BATCH_SIZE= 64
EPOCH_SIZE= 64
TRAIN_TEST_RATIO= 4 # 80/20

# organize dataset from TrainingData.txt file from FDP-Data.py
data_set= SetupDataset(DATA_FILE_PATH, dim= DIM_REDUCTION)
print("Data set Size: %i" % data_set.getSize())

# create training and testing sets
data_set_size= data_set.getSize()
training_set_size= data_set_size - int(data_set_size * (TRAIN_TEST_RATIO/(TRAIN_TEST_RATIO+1)))
testing_set_size= data_set_size - training_set_size
training_set, testing_set= torch.utils.data.random_split(data_set,[training_set_size,testing_set_size])

# create data loader for both training and testing sets
training_data_loader= torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
testing_data_loader= torch.utils.data.DataLoader(testing_set)

# creat the layered model
model= MLModel(data_set.getInputSize(), data_set.getOutputSize())

# train the model
print("\nTraining the model...")
trainModel(model, training_set, training_set_size, EPOCH_SIZE, BATCH_SIZE)

# tests the model and prints result
print("\nTesting the model...")
testModel(model, testing_set, testing_set_size, data_set.getOutputSize())

# try the trained model (currently using a random sample from the data set)
random= int(np.random.uniform(0,data_set.getSize))
sample= data_set.input_data[random]
observed= data_set.output_data[random]


predicted= makePrediciton(sample)
print("Observed Race Results: %i" % observed)

if predicted==observed:
    print("Correct Prediction!")
else:
    print("Incorrect Prediction.")

print("\nDone!")
print("______________________\n")