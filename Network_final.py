import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import keras
from keras import Input, Model, layers
from keras.layers import Flatten
from keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Add

def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

#define residuals
def residual_block(x: tf.Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> tf.Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net(inputs,output_no):

    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    output_cen_eas = layers.Dense(1, activation='sigmoid', name='cen_eas')(t)
    output_cen_wes = layers.Dense(1, activation='sigmoid', name='cen_wes')(t)
    output_bak_sou = layers.Dense(1, activation='sigmoid', name='bak_sou')(t)
    output_bak_nor = layers.Dense(1, activation='sigmoid', name='bak_nor')(t)
    output_vic_sou = layers.Dense(1, activation='sigmoid', name='vic_sou')(t)
    output_vic_nor = layers.Dense(1, activation='sigmoid', name='vic_nor')(t)
    output_no_train = layers.Dense(1, activation='sigmoid', name='no_train')(t)
    
    model = Model(inputs, outputs=[output_cen_eas,output_cen_wes,output_bak_sou,output_bak_nor,output_vic_sou,output_vic_nor,output_no_train])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

#import data
store = pd.HDFStore('initial_datasets/spectro3.h5')
data = store['first']

#output counts of data entries
input_shape = (len(data.array[0]),len(data.array[0][0]))
print('individual input feature size: {} x {}'.format(input_shape[0], input_shape[1]))
"""print('No. trains: {}'.format(data[data.train==1].array.count()))
print('No. cen_eas trains: {}'.format(data[data.train1==1].array.count()))
print('No. cen_wes trains: {}'.format(data[data.train2==1].array.count()))"""

#finalised dataset for training from full dataset
data = pd.concat([data[data.cen_eas==1], data[data.cen_wes==1],data[data.bak_sou==1],data[data.bak_nor==1],data[data.vic_sou==1],data[data.vic_nor==1],data[data.no_train==1].sample(222)])

#randomise rows
data = data.sample(frac=1).reset_index(drop=True)

#convert features and labels into numpy arrays
X = np.array(data.array.tolist())
y0 = np.array(data.no_train.tolist())
y1 = np.array(data.cen_eas.tolist())
y2 = np.array(data.cen_wes.tolist())
y3 = np.array(data.bak_sou.tolist())
y4 = np.array(data.bak_nor.tolist())
y5 = np.array(data.vic_sou.tolist())
y6 = np.array(data.vic_nor.tolist())
#encode the labels
le = LabelEncoder()
yy0 = le.fit_transform(y0)
yy1 = le.fit_transform(y1)
yy2 = le.fit_transform(y2)
yy3 = le.fit_transform(y3)
yy4 = le.fit_transform(y4)
yy5 = le.fit_transform(y5)
yy6 = le.fit_transform(y6)
#split the dataset
#x_train, x_test, y1_train, y1_test, y2_train, y2_test,  = train_test_split(X, yy1, yy2, test_size=0.2, random_state=42)

#set input shape
num_channels = 1
inputs = Input(shape=(input_shape[0],input_shape[1],num_channels), name='spectrogram')

###########################################################################

model = create_res_net(inputs, output_no=6) # or create_plain_net()

model.fit(
    X,
    {"cen_eas": y1, "cen_wes": y2, 'bak_sou':y3, 'bak_nor':y4, 'vic_sou':y5, 'vic_nor':y6, 'no_train':y0},
    epochs=20,
    batch_size=32,
    verbose=2,
    validation_set=0.2
)

keras.utils.plot_model(model, 'networkfinal_model.png', show_shapes=True)

#test_scores = model.evaluate(x_test, y_test, verbose=0)
#print("Test accuracy:", test_scores)

model.save('networkfinal_model.h5')
model.save_weights('networkfinal_weights.h5')