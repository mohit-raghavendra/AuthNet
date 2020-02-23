from keras.applications import inception_v3
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_sp
import numpy as np

number_of_people = ******
desired_size = ******

import pandas as pd

def load_data():
	data_df = pd.read_csv('data.csv')
	
	print(train_df.shape)
	
	data_df.head()
	data_df['person'].value_counts()

	return data_df


print("Called Loaddata")
data_df = load_data()

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im


def preprocess_data(data_df):
    N = data_df.shape[0]
    x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)


    for i, image_id in enumerate((data_df['person'])):
        
        img = preprocess_image('***directory***'+image_id+'.jpg')
        img.save('***directory***'+image_id+'.jpg')


    for i, image_id in enumerate((data_df['person'])):
         

        X[i, :, :, :] = Image.open('****directory*****'+image_id+'.jpg')


	y = pd.get_dummies(data_df['person']).values
    print(X.shape)
    print(y.shape)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=37)
    
    return x_train, x_val, y_train, y_val


print("Called Preprocessing")
x_train, x_val, y_train, y_val = preprocess_data(data_df)




