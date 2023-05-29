from tensorflow import keras
from keras.models import Sequential 
from keras import Input 
from keras.layers import Dense 

def model_ts(input_size):

    model = Sequential(name="Stacked_ann") # Model
    model.add(Input(shape=(input_size,), name='Input-Layer'))
    model.add(Dense(20, activation='sigmoid', name='Hidden-Layer1'))
    model.add(Dense(10, activation='sigmoid', name='Hidden-Layer2')) 
    model.add(Dense(1, activation=None, name='Output-Layer'))

    model.compile(optimizer='adam', 
                loss= keras.losses.MeanSquaredError(),
                )

    return model
