#The final completed source code for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sig
from scipy.stats import ttest_ind
import mne
import random as rand

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#This function takes in a gdf file path and converts it to a pandas dataframe
def path_to_dataframe(FILE_PATH, fs):
    eog = ['EOG-left', 'EOG-central', 'EOG-right']

    #read raw data from gdf using mne
    file_raw = mne.io.read_raw_gdf(FILE_PATH, eog=eog)

    #read events from the gdf annotation
    events, event_id = mne.events_from_annotations(file_raw)
    

    #convert raw to dataframe
    eeg_data = file_raw.to_data_frame()
    eeg_data = eeg_data.drop(eeg_data.columns[0], axis=1)
    eeg_data = eeg_data.drop(eeg_data.columns[-1], axis=1)
    eeg_data = eeg_data.drop(eeg_data.columns[-1], axis=1)
    eeg_data = eeg_data.drop(eeg_data.columns[-1], axis=1)
    eeg_data['event'] = 0


    #convert events to dataframe
    events_df = pd.DataFrame(events)
    events_df = events_df.drop(events_df.columns[1], axis=1)

    #iterates through events and creates a time-matched list to correspond events with eeg timestamps
    track = 0
    hold_size = len(eeg_data)
    hold_list = [0] * hold_size
    for i in events_df[0]:
        time_check = i
        hold_list[i] = events_df.at[track, 2]
        track = track + 1

    #iterates through hold_list (which has markers for the start of trials/ events) and populates a list with labels to associate with eeg timestamps
    one_count = 0
    two_count = 0
    three_count = 0
    four_count = 0
    five_count = 0
    six_count = 0
    seven_count = 0
    eight_count = 0
    nine_count = 0
    ten_count = 0
    hold_two = hold_list.copy()
    for i in range(len(hold_list)):
    #Correspondence of numbers dictionary
    #1023: 1, Rejected trial
    #1072: 2, Eye movements
    #276: 3, Idling EEG (eyes open)
    #277: 4, Idling EEG (eyes closed)
    #32766: 5, Start of a new run
    #768: 6, Start of a trial
    #769: 7, Cue onset left (class 1)
    #770: 8, Cue onset right (class 2)
    #771: 9, Cue onset foot (class 3)
    #772: 10, Cue onset tongue (class 4)
    
        if hold_list[i] == 1:
            #1 indicates a rejected trial (See above "dictionary")
            #Trials are roughly 6 seconds
            #for 250 Hz, this means a trial is approximately 1500 samples in duration

            #Upon seeing a '1', or rejected trial, set the next 6 seconds to '0'
            hold_two[i:(i+(fs*6))] = [0] * (fs*6)
            #Counters to numerate the appearance of each class, primarily for debugging purposes
            one_count = one_count + 1
        elif hold_list[i] == 2:
            #2-4 are irrelevant, also set to '0'
            hold_two[i] = 0
            two_count = two_count + 1
        elif hold_list[i] == 3:
            hold_two[i] = 0
            three_count = three_count + 1
        elif hold_list[i] == 4:
            hold_two[i] = 0
            four_count = four_count + 1
        elif hold_list[i] == 5:
            #5 indicates a new run, starting after a rest period
            #Also irrelevant to classification, so set to '0'
            hold_two[i] = 0
            five_count = five_count + 1
    
        elif hold_list[i] == 6:
            #6 indicates the start of the 6 second trial window
            #Motor Imagery should begin 2 seconds after every 6 second window begins (500 samples)
            #Since each MI task is accurately recorded, this also can be set to '0'
            hold_two[i] = 0
            six_count = six_count + 1
        elif hold_list[i] == 7:
            #7 Indicates left hand MI, or 'class 1'
            #Each motor Imagery task should last ~4 seconds, though the 'end' is not indicated within the data
            #Seeing a 7 will, similarly to rejected trial above, set the next 4 seconds to a value of '1'
            hold_two[i:(i+(fs*4))] = [1] * (fs*4)
            seven_count = seven_count + 1
        if hold_list[i] == 8:
            #8 indicates right hand MI, 'class 2'
            #Set next 4 seconds to '2'
            hold_two[i:(i+(fs*4))] = [2] * (fs*4)
            eight_count = eight_count + 1
        if hold_list[i] == 9:
            #9 - Foot MI
            #Set to '3'
            hold_two[i:(i+(fs*4))] = [3] * (fs*4)
            nine_count = nine_count + 1
        if hold_list[i] == 10:
            #10 - Tongue MI
            #Set to '4'
            hold_two[i:(i+(fs*4))] = [4] * (fs*4)
            ten_count = ten_count + 1

    eeg_data['event'] = hold_two
    return eeg_data

print(f'PyTorch version: {torch.__version__}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print("Done")

#The below method will be given the above devised pandas dataframe from 'path_to_dataframe'
#It will find 20 total samples
#4 each from '0' (Idle), '1', '2', '3', '4' (events, as described above)
#it will return a tensor corresponding to an individual
#This tensor will be used for training of the model, below is provided a method to grab a random test set

def prep_for_ID(df, f, test_ind):
    
    numtimestamp = f * 4
    hold = np.ndarray((20,22,numtimestamp))
    ind = test_ind
    for i in range(4):
        val = df.at[ind, 'event']
        while(val != 0):
            ind = ind + 1
            val = df.at[ind, 'event']
        even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
        hold[i] = even.T
        ind = ind + numtimestamp
    ind = test_ind
    for i in range(4):
        val = df.at[ind, 'event']
        while(val != 1):
            ind = ind + 1
            val = df.at[ind, 'event']
        even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
        hold[i+4] = even.T
        ind = ind + numtimestamp
    ind = test_ind
    for i in range(4):
        val = df.at[ind, 'event']
        while(val != 2):
            ind = ind + 1
            val = df.at[ind, 'event']
        even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
        hold[i+8] = even.T
        ind = ind + numtimestamp
    ind = test_ind
    for i in range(4):
        val = df.at[ind, 'event']
        while(val != 3):
            ind = ind + 1
            val = df.at[ind, 'event']
        even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
        hold[i+12] = even.T
        ind = ind + numtimestamp
    ind = test_ind
    for i in range(4):
        val = df.at[ind, 'event']
        while(val != 4):
            ind = ind + 1
            val = df.at[ind, 'event']
        even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
        hold[i+16] = even.T
        ind = ind + numtimestamp
    #tens = torch.from_numpy(hold)
    tens = torch.FloatTensor(hold)
    return tens

#This method grabs a single sample at the specified index (as a tensor)
#for testing of model
def test_tens_ID(df, f, test_ind):
    numtimestamp = f * 4
    hold = np.ndarray((1,22,numtimestamp))
    ind = test_ind
    even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
    hold[0] = even.T
    tens = torch.FloatTensor(hold)
    return tens

def test_tens_ID_2(df, f):
    numtimestamp = f * 4
    hold = np.ndarray((1,22,numtimestamp))
    ind = rand.randint(10000, 600000)
    #print("Randint:", ind)
    even = df.iloc[ind:(ind+numtimestamp),0:22].to_numpy()
    hold[0] = even.T
    tens = torch.FloatTensor(hold)
    return tens

class IDTest_n(nn.Module):
    def __init__(self, n_channels=22, time_steps=1000, n_subj=4):
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv1d(in_channels=22, out_channels=32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_subj), #Outputs a (n, 1) tensor
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

#Initialize files as dataframes
FILE_PATH_1 = 'C:/Users/semee/Class/EEEE 536/Project/Data/A01T.gdf'
FILE_PATH_2 = 'C:/Users/semee/Class/EEEE 536/Project/Data/A03T.gdf'
FILE_PATH_3 = 'C:/Users/semee/Class/EEEE 536/Project/Data/A05T.gdf'
FILE_PATH_4 = 'C:/Users/semee/Class/EEEE 536/Project/Data/A06T.gdf'
freq = 250
numsubj = 4
n_samples = 20
n_channels = 22
time_steps = freq*4
subj1 = path_to_dataframe(FILE_PATH_1, freq)
subj2 = path_to_dataframe(FILE_PATH_2, freq)
subj3 = path_to_dataframe(FILE_PATH_3, freq)
subj4 = path_to_dataframe(FILE_PATH_4, freq)

#Convert dataframes to single tensor
ind = 100000
tensor_x_1 = prep_for_ID(subj1, freq, ind)
tensor_x_2 = prep_for_ID(subj2, freq, ind)
tensor_x_3 = prep_for_ID(subj3, freq, ind)
tensor_x_4 = prep_for_ID(subj4, freq, ind)

x = torch.cat([tensor_x_1, tensor_x_2, tensor_x_3, tensor_x_4], dim=0)
hold = np.zeros((n_samples*numsubj, numsubj))
rownum = 0
for i in range(numsubj):
    hold[rownum:rownum+n_samples, i] = 1
    rownum = rownum + n_samples
y = torch.FloatTensor(hold)

#Run n_subject identifier
model = IDTest_n(n_channels, time_steps, numsubj)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

subjects = [subj1, subj2, subj3, subj4]
signals = torch.FloatTensor([])
train = []
val = []
x_ax = []
y_test = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
epoch = 1000
for i in subjects:
    signals = torch.cat([signals, test_tens_ID_2(i, freq)], dim=0)
for i in range(epoch):
    model.train()
    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        train.append(loss.item())
        model.eval()
        val_loss_hold = 0
        with torch.no_grad():
            prob = model(signals)
            val_loss = criterion(prob, y_test)
        val.append(val_loss)
        x_ax.append(i+1)
plt.plot(x_ax, train)
plt.plot(x_ax, val)
plt.xlabel("Epoch")
plt.ylabel("Loss")
labels = ["Training Loss", "Value Loss"]
plt.legend(labels)
plt.show()