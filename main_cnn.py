# Authors: Huaguo Chen, Xinhong Chen
# Contact: xinhong.chen@cityu.edu.hk

from data import Dataset, DataGenerator
import keras
import json
import numpy as np
import os
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from model import CNN_Model
import argparse


# the data generator, which generates a batch of data for training or testing
class data_generator_cnn(DataGenerator):
    def __iter__(self, random=False):
        x, y = [], []

        for is_end, i in self.sample(random):
            x.append(self.data[i])
            y.append(self.label[i])
            if len(y) == self.batch_size or is_end:
                yield np.array(x), np.array(y)
                x, y = [], []


parser = argparse.ArgumentParser(description="for_main")
parser.add_argument(
    '-m',
    "--modelname",
    default='cnn',
    help="cnn, threeparts",
    type=str,
)

args = parser.parse_args()

# read the dataset
config = json.load(open('config.json', 'r'))
data = Dataset(config)

X, Y = data.data, data.label
Ymax, Ymin = data.labelmax, data.labelmin

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists('log'):
    os.makedirs('log')
if not os.path.exists('res'):
    os.makedirs('res')

res = []
# get 10 folds of data, and do the training-testing process for 10 folds.
kf = KFold(n_splits=10, shuffle=True, random_state=0)
for train_index, test_index in kf.split(Y):
    # get the training and testing data in this fold
    train_features = X[train_index]
    train_label = Y[train_index]

    test_features = X[test_index]
    test_label = Y[test_index]

    train_generator = data_generator_cnn(config['batch_size'], train_features, train_label, random=True)
    test_generator = data_generator_cnn(config['batch_size'], test_features, test_label)

    model = CNN_Model(config).model

    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.01)
    )

    # the evaluation function for any given data
    def evaluate(input_data):
        # the calculation process of all performance indicators
        all_y_true = []
        all_y_pred = []
        for x_true, y_true in input_data:
            y_pred = model.predict(x_true)
            y_pred_ori = y_pred * Ymax + (1-y_pred) * Ymin
            y_true_ori = y_true * Ymax + (1-y_true) * Ymin
            all_y_pred += list(y_pred_ori.flatten())
            all_y_true += list(y_true_ori.flatten())

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        re = abs(all_y_pred - all_y_true) / all_y_true

        R_1 = np.sum((all_y_true - all_y_true.mean()) * (all_y_pred - all_y_pred.mean()))
        R_2 = np.sqrt(np.sum((all_y_pred - all_y_pred.mean()) ** 2) * np.sum((all_y_true - all_y_true.mean()) ** 2))
        R = R_1 / (R_2 + 1e-5)

        rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))

        mae = mean_absolute_error(all_y_true, all_y_pred)

        rrmse = (rmse / all_y_true.mean()) * 100

        rmae = (mae / all_y_true.mean()) * 100

        return [R, rmse, mae, rrmse, rmae, re.mean()]


    class Evaluator(keras.callbacks.Callback):
        def __init__(self):
            self.best_epoch = 0
            self.best_rmse = 99999.9

        def on_epoch_end(self, epoch, logs=None):
            _, rmse, _, _, _, _ = evaluate(test_generator)
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_epoch = epoch
                model.save_weights('checkpoints/%s_best_model.weights' % args.modelname)


    csv_logger = CSVLogger('log/%s_training_fold%d.log' % (args.modelname, len(res)))

    evaluator = Evaluator()

    # fit the model with the training generator
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=500,
        callbacks=[evaluator, csv_logger]
    )

    # load the best model weights during training
    model.load_weights('checkpoints/%s_best_model.weights' % args.modelname)

    # get the test performance of fold i
    tmp = evaluate(test_generator)
    res.append(tmp)
    print(tmp)

    # delete model for efficient memory reusage
    K.clear_session()
    del model

# collect all the testing performances and save them, and meanwhile calculate their mean performance
res = np.array(res)
# print(res.mean(axis=0))
with open('res/%s_res.txt' % args.modelname, 'w') as f:
    for line in res:
        f.write(','.join([str(i) for i in line]) + "\n")
    f.write(','.join([str(i) for i in res.mean(axis=0)])+"\n")