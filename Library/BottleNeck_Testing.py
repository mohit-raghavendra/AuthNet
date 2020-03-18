from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from PIL import Image

from numpy import save

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution1D, Convolution2D, ZeroPadding2D, MaxPooling2D

import os

trial_model=load_model('C:\\Users\\Om\\Desktop\\Cerberus\\Utilities\\VGGFace.h5')
WB_CNN = load_model('C:\\Users\\Om\\Desktop\\Cerberus\\Library\\Bottle_Neck.h5')



def features(featmodel, crpimg, transform=False): #extract face feature vector
    
    # transform=True seems more robust but I think the RGB channels are not in right order
    
    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:,:,0] -= 129.1863
        imarr[:,:,1] -= 104.7624
        imarr[:,:,2] -= 93.5940        
        aux = copy.copy(imarr)
    imarr = np.expand_dims(imarr, axis=0)
    output= featmodel.predict(imarr)[0,:]
    return output


def feature_extractor(inpDir,person,timestep,sample,ex_type=True): #access directory and extract face vector
    """
    Funtion that extracts the face features into numpy arrays
    
    Parameters: 
    featmodel: model used for prediction. VGGFace in this case.
    crpimg: variable that stores the images and correct RGB values.
    
    Returns: 
    output : features of the image passed
    """    
    
     
    if ex_type == True:
        Y_train_labels=np.ones((1,2622)) #*******
    else:
        Y_train_labels=np.zeros((1,2622))  #remove after training stage is completed  **********
        
     
    sample_space=[] #for appending different utterances
    feature_sequence = [] # for appending different timesteps
     #input dataset most likely to be People_cerebrus/photos
    linpDir = os.listdir(inpDir) #list all directories in dataset
    #print(linpDir)
    personStr= linpDir[person]
    sampleFolder = '%s\\%s' % (inpDir,personStr) #opening sample folder
    lsampleFolder = os.listdir(sampleFolder)
    print(lsampleFolder)
    sample= lsampleFolder[sample]
    
    utterFolder = '%s\\%s' % (sampleFolder,sample)  #opening utterance folder
    i = 0
    lutterFolder = os.listdir(utterFolder) 
    
    for utterances in lutterFolder:
        print(utterances)
        utterNumber= '%s\\%s' % (utterFolder,utterances)
        lutterNumber= os.listdir(utterNumber)

        frame = lutterNumber[timestep] #accessing images of required timestep
        i = i + 1
        image= "%s\\%s" % (utterNumber,frame) 
        im=Image.open(image)
        im = im.resize((224,224))
        feature_vector = features(trial_model,im, transform=True).reshape((1,1, 1, 2622))  
        if i==1 :
            feature_sequence=feature_vector #done because of need of same dim. for concatenation
        else:
            feature_sequence = np.concatenate((feature_sequence,feature_vector),axis=0)

    print("Extraction completed %d"%i)
    
    feature_sequence1 = feature_sequence[:,0,:,:]
    return feature_sequence1  

def word_extractor(inpDir,person,sample,ex_type):
        
    """
    Function that extracts face features of all images across all timesteps for one word.
    
    Parameters:
    inpDir: String that stores path of photos directory
    person: variable that stores the person number **not a string value**
    sample: variable that stores the sample number(sample is the word which is spoken) **not a string value**
    
    Returns:
    CNN_out_total: Features of all utterances of the sample/word across all the timesteps
    
    """
    timesteps=20 
    count=0
    for i in range(timesteps):
        CNN_out=feature_extractor(inpDir,person,i,sample,ex_type)
        count = count + 1
        if count==1 :
            CNN_out_total= CNN_out #maintaining same dimensions
        else :
            CNN_out_total= np.concatenate((CNN_out_total,CNN_out))
        print(str(i) + 'timestep')
        
    print('Concatenation completed')
    return CNN_out_total     

def person_extractor(InpDir,person):
        
    """
    Function that extracts face features of all images for the person.
    
    Parameters:
    inpDir: variable that stores path of photos directory
    person: variable that stores the person number **not a string value**
    
    Returns:
    p_dataset: list that has the features extracted for all the videos of the person
    
    """
    
    p_dataset=[]
    no_of_words=3
    count=0
    for i in range(no_of_words): #Iterate through all the words spoken by the person
        count = count +1
        if person==0 and i==0:
            CNN_out_total= word_extractor(InpDir,person,i,True)
        else:
            CNN_out_total= word_extractor(InpDir,person,i,False)
        if count==1 :
            p_dataset = CNN_out_total #done because of need of same dim. for concatenation
        else:
            p_dataset = np.concatenate((p_dataset,CNN_out_total),axis=0)
        print(str(i) + 'word')
        print('Printing Dataset')
        print(p_dataset)
    print('----------------')
    print(count)
    print(p_dataset.shape)
    return p_dataset 

def reshape_LSTM(sample_mat):
    
    """
    Function to reshape the face features array into a LSTM type input

    Parameters:
    sample_mat: list that has the dataset of extracted features for 1 person
    
    Returns:
    LSTM_input: list that has the image features in the appropriate sequence so that it can be reshaped to LSTM input format
    
    """
    feature_vector=[]
    feature_mat=[]
    
    word_no = 3
    utterance_no = 1
    
    count_word = 0
    for word in range(word_no):  #Go through the number of words
        count_i=0
        for utterance in range(utterance_no):  #Go through the number of utterances
            count_j=0
            #Take 100 images/features_sequences(each word has 5 utterances * 20 timesteps = 100 images/features_sequences)
            for timestep in range((20*utterance_no)*word,(20*utterance_no)*(word+1),utterance_no): 
                #The images go through each utterance of 1 timestep and then go to the next timestep and repeat
                #So, we take every fifth image and concatenate(which gives utterance*20/utterance =  20 images i.e all the timesteps of 1 utternace) 
                 
                timestep = timestep + utterance
                sample_mat_j=sample_mat[timestep]
                sample_mat_j=sample_mat_j.reshape((1,2622))
                if count_j==0:
                    feature_vector=sample_mat_j
                else:
                    feature_vector=np.concatenate((feature_vector,sample_mat_j))
                count_j = count_j + 1

            if count_i==0:
                feature_mat=feature_vector
            else:
                feature_mat= np.concatenate((feature_mat,feature_vector))
                #This has the features for each timestep for 1 utterance. This is repeated for all utternaces in the outer loop
            count_i= count_i + 1
            print(feature_mat.shape)
            
        if count_word==0:
            LSTM_input=feature_mat
        else:
            LSTM_input= np.concatenate((LSTM_input,feature_mat))
            #This has all the features for 1 word. This is repeated for each word in the outer loop
        count_word= count_word + 1
                
    print(LSTM_input.shape)
  
    return LSTM_input

def BottleNeck(InpDir):
    """
    Function that returns the dataset for training the model.
    
    Parameters:
    inpDir: variable that stores path of photos directory
    
    Returns:
    p_total_dataset: Dataset that has the features of all the videos of all the people
    """
    
    p_dataset=person_extractor(InpDir,0)
    p_dataset_LSTM=reshape_LSTM(p_dataset)
    print('person done')
    return p_dataset_LSTM
