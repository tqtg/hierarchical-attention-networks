import pickle
from tqdm import tqdm
import random
import numpy as np

class DataReader:
  def __init__(self, train_file, dev_file, test_file,
               max_word_length=50, max_sent_length=30, num_classes=5):
    self.max_word_length = max_word_length
    self.max_sent_length = max_sent_length
    self.num_classes = num_classes

    self.train_data = self._read_data(train_file)
    self.valid_data = self._read_data(dev_file)
    self.test_data = self._read_data(test_file)

  def _read_data(self, file_path):
    print('Reading data from %s' % file_path)
    new_data = []
    with open(file_path, 'rb') as f:
      data = pickle.load(f)
      random.shuffle(data)
      for label, doc in data:
        doc = doc[:self.max_sent_length]
        doc = [sent[:self.max_word_length] for sent in doc]

        label -= 1
        assert label >= 0 and label < self.num_classes

        new_data.append((doc, label))

    # sort data by sent lengths to speed up
    new_data = sorted(new_data, key=lambda x: len(x[0]))
    return new_data

  def _batch_iterator(self, data, batch_size, desc=None):
    num_batches = int(np.ceil(len(data) / batch_size))
    for b in tqdm(range(num_batches), desc):
      begin_offset = batch_size * b
      end_offset = batch_size * b + batch_size
      if end_offset > len(data):
        end_offset = len(data)

      doc_batch = []
      label_batch = []
      for offset in range(begin_offset, end_offset):
        doc_batch.append(data[offset][0])
        label_batch.append(data[offset][1])

      yield doc_batch, label_batch

  def read_train_set(self, batch_size, shuffle=False):
    if shuffle:
      random.shuffle(self.train_data)
    return self._batch_iterator(self.train_data, batch_size, desc='Training')

  def read_valid_set(self, batch_size):
    return self._batch_iterator(self.valid_data, batch_size, desc='Validating')

  def read_test_set(self, batch_size):
    return self._batch_iterator(self.test_data, batch_size, desc='Testing')
