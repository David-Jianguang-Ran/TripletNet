import pandas as pd
import numpy as np
import matplotlib as plt

import code
import random


from data_generator import MnistTripletGenerator

# mnist_data = pd.read_csv("./mnist/train.csv")
#
# some_label = random.randint(0,9)
# same = mnist_data.query(f'label == {some_label}')
# not_same = mnist_data.query(f'label != {some_label}')
#
# same.drop(columns='label')
#
# block_0 = np.zeros([100,1])

# open interactive shell
code.interact(local=locals())
