from __future__ import absolute_import, division, print_function
# standard numerical library imports

# energyflow imports
import energyflow as ef
from energyflow.archs import LinearClassifier
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, standardize, to_categorical



import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import torch.nn.init as init
from torch.autograd import Variable

def main():
    f = pd.read_hdf("/data/t3home000/spark/LHCOlympics/data/events_anomalydetection.h5")
    dt = f.values
    dt2 = dt[:,:2100]
    dt3 = dt[:,-1]

    dt2 = dt2.reshape((len(dt2), len(dt2[0])//3, 3))

    # data controls
    num_data = 1100000
    test_frac = 0.2

    # efp parameters
    dmax = 7
    measure = 'hadr'
    beta = 0.5

    print('Calculating d <= {} EFPs for {} jets... '.format(dmax, num_data), end='')
    efpset = ef.EFPSet(('d<=', dmax), measure='hadr', beta=beta)

    test = dt2[:1000]
    test_X = [x[x[:,0] > 0] for x in test]
    X = efpset.batch_compute(test_X)
    print(X)
    print('done')

if __name__ == '__main__':
    main()

