import pandas as pd
import pyarrow.parquet as pq # Used to read the data
import os 
import numpy as np
import tensorflow as tf
from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model, load_model
from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split 
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting

# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32) 

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# This function standardize the data from (-128 to 127) to (-1 to 1)
# Theoretically it helps in the NN Model training, but I didn't tested without it
def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:    
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]

def transform_ts(ts, n_dim=160, min_max=(-1,1)):
    # convert data into -1 to 1
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),percentil_calc, relative_percentile]))

    return np.asarray(new_ts)


# select how many folds will be created
N_SPLITS = 5
# it is just a constant with the measurements data size
sample_size = 800000

# in other notebook I have extracted the min and max values from the train data, the measurements
max_num = 127
min_num = -128


X = np.load('../input/X.npy')
y = np.load('../input/y.npy')
print(X.shape, y.shape)


def data_normalization(train, test):
    train = train.astype('float64')
    test = test.astype('float64')

    mean = train.mean()
    train -= mean
    std = train.std()
    train /= std

    test -= mean
    test /= std

    return train, test

def load_selected_features(col):
    train_file = "train_featuresHiLo_thresh_4.5_db4.csv"
    test_file = "test_featuresHiLo_thresh_4.5_db4.csv"
    if not (os.path.isfile('../input/' + train_file) \
            & os.path.isfile('../input/' + test_file)):
        print("No features files..")
        raise AssertionError()

    features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "low_high_ratio", "hi_true", "lo_true", "low_high_ratio_true"]
    target = ["fault"]

    selected = col
    data_file = "../input/" + train_file
    df_train = pd.read_csv(data_file)
    # train = df_train[features + target]
    train = df_train[selected]

    data_file = "../input/" + test_file
    df_test = pd.read_csv(data_file)
    # test = df_test[features]
    test = df_test[selected]

    # data normalization
    train, test = data_normalization(train.values, test.values)

    return train, test


train_feature, test_feature = load_selected_features('num_true_peaks')

print("train shape : {} , test shape : {}".format(train_feature.shape, test_feature.shape))

# This is NN LSTM Model creation
def model_lstm(input_shape):
    # The shape was explained above, must have this order
    inp = Input(shape=(input_shape[1], input_shape[2],))
    # This is the LSTM layer
    # Bidirecional implies that the 160 chunks are calculated in both ways, 0 to 159 and 159 to zero
    # although it appear that just 0 to 159 way matter, I have tested with and without, and tha later worked best
    # 128 and 64 are the number of cells used, too many can overfit and too few can underfit
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)
    # The second LSTM can give more fire power to the model, but can overfit it too
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    # Attention is a new tecnology that can be applyed to a Recurrent NN to give more meanings to a signal found in the middle
    # of the data, it helps more in longs chains of data. A normal RNN give all the responsibility of detect the signal
    # to the last cell. Google RNN Attention for more information :)
    x = Attention(input_shape[1])(x)
    # A intermediate full connected (Dense) can help to deal with nonlinears outputs
    x = Dense(64, activation="relu")(x)
    
    x = Dense(32, activation="relu")(x)

    # A binnary classification as this must finish with shape (1,)
    # x = Dense(1, activation="sigmoid")(x)

    # second input
    inp2 = Input(shape=(1,))
    x2 = Dense(32, activation="relu")(inp2)

    out = Add()([x, x2])

    out = Dense(1, activation="sigmoid")(out)

    model = Model(inputs=[inp, inp2], outputs=out)
    # Pay attention in the addition of matthews_correlation metric in the compilation, it is a success factor key
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
    
    return model

model_dir = "add_true_peak"

splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))
preds_val = []
y_val = []
# Then, iteract with each fold
# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
for idx, (train_idx, val_idx) in enumerate(splits):
    K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)
    print("Beginning fold {}".format(idx+1))
    # # use the indexes to extract the folds in the train and validation data
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    # selected feature split
    tr_feature_X, val_feature_X = train_feature[train_idx], train_feature[val_idx]

    # # instantiate the model for this fold
    model = model_lstm(train_X.shape)
    # # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
    # # validation matthews_correlation greater than the last one.
    # ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')
    # # Train, train, train
    # model.fit(train_X, train_y, batch_size=128, epochs=50, validation_data=[val_X, val_y], callbacks=[ckpt])
    # # loads the best weights saved by the checkpoint
    model.load_weights('../model/{}/weights_{}.h5'.format(model_dir, idx))
    # Add the predictions of the validation to the list preds_val
    preds_val.append(model.predict([val_X, val_feature_X], batch_size=512))
    # and the val true y
    y_val.append(val_y)


# concatenates all and prints the shape    
preds_val = np.concatenate(preds_val)[...,0]
y_val = np.concatenate(y_val)

print("{}, {}".format(preds_val.shape, y_val.shape))

# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result

best_threshold = threshold_search(y_val, preds_val)['threshold']

preds_val = (preds_val > best_threshold).astype(np.int)
print("preds_val_shape : {}".format(preds_val.shape))

val_score = (preds_val == y_val).sum() / y_val.shape[0]
print("val_socre : {}".format(val_score))

submission_title = "{}_{}".format(round(val_score, 3), best_threshold)

meta_test = pd.read_csv('../input/metadata_test.csv')
meta_test = meta_test.set_index(['signal_id'])


def transform_test():

     # First we daclarete a series of parameters to initiate the loading of the main data
    # it is too large, it is impossible to load in one time, so we are doing it in dividing in 10 parts
    first_sig = meta_test.index[0]
    n_parts = 10
    max_line = len(meta_test)
    part_size = int(max_line / n_parts)
    last_part = max_line % n_parts
    print(first_sig, n_parts, max_line, part_size, last_part, n_parts * part_size + last_part)
    # Here we create a list of lists with start index and end index for each of the 10 parts and one for the last partial part
    start_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]
    start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]
    print(start_end)

    X_test = []
    # now, very like we did above with the train data, we convert the test data part by part
    # transforming the 3 phases 800000 measurement in matrix (160,57)
    for start, end in start_end:
        subset_test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
        for i in tqdm(subset_test.columns):
            id_measurement, phase = meta_test.loc[int(i)]
            subset_test_col = subset_test[i]
            subset_trans = transform_ts(subset_test_col)
            X_test.append([i, id_measurement, phase, subset_trans])

    X_test_input = np.asarray([np.concatenate([X_test[i][3],X_test[i+1][3], X_test[i+2][3]], axis=1) for i in range(0,len(X_test), 3)])
    np.save("X_test.npy",X_test_input)

    return X_test_input

if (os.path.isfile('./X_test.npy')) :
    X_test_input = np.load('./X_test.npy')
else:
    X_test_input = transform_test()

print("X_test_input shape : {}".format(X_test_input.shape))

submission = pd.read_csv('../input/sample_submission.csv')
print(len(submission))

preds_test = []
for i in range(N_SPLITS):
    model.load_weights('../model/{}/weights_{}.h5'.format(model_dir, i))
    pred = model.predict([X_test_input, test_feature], batch_size=300, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(3):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)

preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)
print("preds_test_shape : {}".format(preds_test.shape))

submission['target'] = preds_test

import time
curtime = time.strftime('%m%d%H%M%S')

submission.to_csv('../submissions/{}_{}_{}.csv'.format(submission_title, model_dir, curtime), index=False)