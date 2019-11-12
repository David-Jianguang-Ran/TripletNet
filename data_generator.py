
import pandas as pd
import numpy as np
import math

from keras.utils import Sequence


class MnistSingleGenerator(Sequence):
  def __init__(self, batch_size,data_path):
    self.batch_size = batch_size
    self.data_source = pd.read_csv(data_path)

  def __len__(self):
    return len(self.data_source) // self.batch_size

  def __getitem__(self, index):
    sample = self.data_source.sample(self.batch_size)
    sample = sample.reset_index().drop(columns=['index'])

    y = sample['label']
    x = sample.drop(columns=['label'])
    return x,y


class MnistDoubleGenerator(Sequence):
  def __init__(self,batch_size, data_path):
    self.batch_size = batch_size
    self.data_source = pd.read_csv(data_path)

  def __len__(self):
    return len(self.data_source) // self.batch_size

  def __getitem__(self, index):
    def encode_same_label(row):
      if row[0] == row[257]:
        return 1.0
      else:
        return 0.0

    left = self.data_source.sample(self.batch_size)
    left = left.reset_index().drop(columns=['index'])
    right = self.data_source.sample(self.batch_size)
    right = right.reset_index().drop(columns=['index'])

    together = pd.concat([left,right],axis=1, join="outer", ignore_index=True, sort=False)

    x = together.drop(columns=[0,257])
    y = together.apply(encode_same_label,axis=1)

    return x, y



class MnistTripletGenerator(Sequence):
  """
  
  soooo how do we want to do this?
  
  approach one:

  for each number in between 0 and 9 ( include )
  filter by some label number, get a df of just one number
  example :
  same = df.query('label == {}')
  not_same = df.query('label != {})
    note: I guess @varname also works in query string
    but in the interest of readability lets keep it plain python (f string)

  sample batch_size / 10 samples from the same group and make anchor and positive
  sample same number of not_same and make it negative
  
  voila, here is our triplet dataset
  (make sure each batch contains each hand written number)
  
  """
  def __init__(self, batch_size, data_path, *args, **kwargs):
    self.batch_size = batch_size
    self.data_source = pd.read_csv(data_path)
    
  def __len__(self):
    '''
    :return: int number of batches
    '''
    return len(self.data_source) // self.batch_size

  def __getitem__(self, index):
    """

    :param index:
    :return: (x , y) x is a concat of the triplet order = (a,p,n),
    y is a array of zeros just to keep keras happy
    """
    # hard coded shape for three mnist img concated together
    x = np.empty((0, 2352))
    for anchor_num in range(0,10):
      same = self.data_source.query(f'label == {anchor_num}').drop(columns='label')
      not_same = self.data_source.query(f'label != {anchor_num}').drop(columns='label')

      anchor = same.sample(self.batch_size // 10)
      positive = same.sample(self.batch_size // 10)
      negative = not_same.sample(self.batch_size // 10)

      data_each_num = np.concatenate((anchor, positive, negative), axis=1)
      x = np.concatenate((x, data_each_num), axis=0)

    y = np.zeros((len(x), 1))

    return x, y
