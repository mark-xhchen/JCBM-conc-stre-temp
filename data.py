# Authors: Huaguo Chen, Xinhong Chen
# Contact: xinhong.chen@cityu.edu.hk

import numpy as np
import os, json

"""
0: Temperature
1: Specimens
2: Heating rate
3: Maintenance
4: Cement
5: Water
6: Sand
7: Aggregate
8: Silica Fun1e
9: Fly ash
10: Slag
11: Mrtakalin
12: S1eel fiber
13: SF_Length
14: SF_Aspect ratio were
15: Polymer fiber
16: PF_Length
17: PF_Aspect ratio were
18: Superplasticizer
"""


class Dataset():
    def __init__(self, config):
        self.config = config

        self.dtype = [('Temperature', np.int64), ('Heating_Rate', float), ('Maintenance', float), 
                      ('Specimens', np.int64), ('Cement', float), ('Water', float), ('Sand', float), ('Aggregate', float),  
                      ('Silica',float), ('Flyash', float), ('Slag', float), ('Mrtakalin', float), ('Superplasticizer', float),
                      ('Steel_Fiber', float), ('SF_Length', float), ('SF_ARW', float), ('Polymer_Fiber', float), ('PF_Length', float), ('PF_ARW', float)]

        self.oridata, self.orilabel = self.read_data(self.config['file_path'])
        self.data, self.label, self.labelmax, self.labelmin = self.preprocess()

    def read_data(self, path):
        ret = []
        label = []
        with open(path, 'r') as csvfile:
            for line in csvfile:
                #line = line.replace('\ufeff800', "800")
                tmp = line.strip().split(',')
                ret.append(tuple([float(tmp[i]) for i in range(len(tmp)-1)]))
                label.append(float(tmp[-1]))

        return np.array(ret, dtype=self.dtype), np.array(label)

    def preprocess(self):
        ret = []

        data = np.array([list(i) for i in self.oridata])
        
        # conduct max-min normalization on data 
        feature_max = []
        feature_min = []
        for i in range(data.shape[1]):
            tmpmax = data[:, i].max()
            tmpmin = data[:, i].min()
            data[:, i] = (data[:, i] - tmpmin) / (tmpmax - tmpmin)

        # conduct the same normalization on targeted values
        labelmax = self.orilabel.max()
        labelmin = self.orilabel.min()
        label = (self.orilabel - labelmin) / (labelmax - labelmin)

        return data, label, labelmax, labelmin


class DataGenerator(object):
    def __init__(self, batch_size, data, label, random=False):
        self.batch_size = batch_size
        self.data = data
        self.label = label
        self.steps = len(self.label) // self.batch_size
        if len(self.label) % self.batch_size != 0:
            self.steps += 1
        self.random = random

    def __len__(self):
        return self.steps

    def sample(self, random):
        if random:
            def random_gen():
                indices = list(range(len(self.label)))
                np.random.shuffle(indices)
                for i in indices:
                    yield i

            data_gen = random_gen()
        else:
            data_gen = iter(list(range(len(self.label))))

        d_current = next(data_gen)
        for d_next in data_gen:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d


if __name__ == "__main__":
    config = json.load(open('json/config.json', 'r'))
    data = Dataset(config)

    #print(data.data.shape, data.label.shape)
    print(data.data)