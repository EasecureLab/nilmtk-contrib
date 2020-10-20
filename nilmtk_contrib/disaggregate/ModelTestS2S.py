from __future__ import print_function, division
from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
import pandas as pd
import numpy as np
from collections import OrderedDict
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

class ModelTestS2S(Disaggregator):

    def __init__(self, params):
        self.MODEL_NAME = "ModelTestS2S"
        self.models = OrderedDict()

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')
        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, 99, 1))
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, 99))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            self.models[appliance_name] = self.return_network()
            model = self.models[appliance_name]
            train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
            model.fit(train_x, train_y, validation_data=[v_x, v_y], epochs=5, batch_size=512)

    def disaggregate_chunk(self, test_main_list, model=None):
        test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')
        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, 99, 1))
            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                prediction = model.predict(test_main_array, batch_size=512)
                n = len(prediction) + 99 - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                for i in range(len(prediction)):
                    sum_arr[i:i + 99] += prediction[i].flatten()
                    counts_arr[i:i + 99] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]
                prediction = 51 + (sum_arr * 83)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):
        model = Sequential()
        model.add(Conv1D(30, 10, activation="relu", input_shape=(99, 1), strides=2))
        model.add(Conv1D(30, 8, activation='relu', strides=2))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(99))
        model.compile(loss='mse', optimizer='adam')
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        processed_mains_lst = []
        for mains in mains_lst:
            new_mains = mains.values.flatten()
            new_mains = np.pad(new_mains, (49, 49), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + 99] for i in range(len(new_mains) - 99 + 1)])
            new_mains = (new_mains - 51) / 83
            processed_mains_lst.append(pd.DataFrame(new_mains))
        if method == 'train':
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (49, 49), 'constant', constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + 99] for i in range(len(new_app_readings) - 99 + 1)])
                    new_app_readings = (new_app_readings - 51) / 83
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_app_dfs))
            return processed_mains_lst, appliance_list
        else:
            return processed_mains_lst