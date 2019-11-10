import pandas as pd
import numpy as np
import matplotlib as plt

import code
import random


from data_generator import MnistTripletGenerator

encoding = pd.read_csv("./mnist/fashion_precomp_encoderIII_test.csv")
#
# some_label = random.randint(0,9)
# same = mnist_data.query(f'label == {some_label}')
# not_same = mnist_data.query(f'label != {some_label}')
#
# same.drop(columns='label')
#
# block_0 = np.zeros([100,1])

a = encoding.sample(frac=1)
b = encoding.sample(frac=1)

together = pd.concat([a,b],axis=1,join="outer",ignore_index=True)

def same_label(row):
    if row[0] == row[257]:
        return 1.0
    else:
        return 0.0

# open interactive shell
labels = together.apply(same_label)
output = together.drop(columns=[0,257])
code.interact(local=locals())
