import pandas as pd
import numpy as np

from keras.utils import Sequence


class MnistDataGenerator(Sequence):
  def __init__(self, batch_size,data_path, batch_count=None, *args,**kwargs):
    self.batch_size = batch_size
    self.data_source = pd.read_csv(data_path)
    if batch_count:
      self.batch_count = batch_count
    self.formatted = None

  def __len__(self):
    return len(self.data_source) // self.batch_size

  def __getitem__(self, item):
    raise NotImplementedError

  def get_sample_clean_index(self):
    sample = self.data_source.sample(self.batch_size)
    return sample.reset_index(drop=True)


class MnistSingleGenerator(MnistDataGenerator):
  def __getitem__(self, index):
    sample = self.get_sample_clean_index()

    y = sample['label']
    x = sample.drop(columns=['label'])
    return x,y


class MnistDoubleGenerator(MnistDataGenerator):

  def _make_formatted_data(self):
    """
    returns pair of encodings with label 1.0 or 0.0 for same / diff all in one df
    :return:
    """
    def encode_same_label(row):
      if row[0] == row[257]:
        return 1.0
      else:
        return 0.0

    left = self.data_source.sample(frac=1)
    left = left.reset_index(drop=True)

    right = self.data_source.sample(frac=1)
    right = right.reset_index(drop=True)

    together = pd.concat([left,right],axis=1, join="outer", ignore_index=True, sort=False)
    together['label'] = together.apply(encode_same_label,axis=1)
    together = together.sample(frac=1).reset_index().drop(columns=[0,257,'index'])

    return together

  def __len__(self):
    return len(self.data_source) // self.batch_size

  def __getitem__(self, index):
    if type(self.formatted) == type(None):
      self.formatted = self._make_formatted_data()

    positive_cases = self.formatted.query("label == 1.0").sample(n=self.batch_size // 2,replace=True)
    negative_cases = self.formatted.query("label == 0.0").sample(n=self.batch_size // 2)

    together = pd.concat([positive_cases,negative_cases], axis=0, ignore_index=True)
    y = together['label']
    x = together.drop(columns=['label'])

    return x , y

  def on_epoch_end(self):
    self.formatted = self._make_formatted_data()


class MnistTripletGenerator(MnistDataGenerator):
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


class MnistDemoGenerator(MnistDataGenerator):
    def __len__(self):
      return self.batch_count

    def __getitem__(self, index):
      def encode_same_label(row):
        if row.iloc[0] == row.iloc[1]:
          return 1.0
        else:
          return 0.0

      left = self.get_sample_clean_index()
      right = self.get_sample_clean_index()

      label_raw = pd.concat([left['label'],right['label']],axis=1)
      y = label_raw.apply(encode_same_label,axis=1)

      return left.drop(columns=['label']).to_numpy(), right.drop(columns=['label']).to_numpy(), y

