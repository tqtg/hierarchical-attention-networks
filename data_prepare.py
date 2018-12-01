import pandas as pd
import nltk
import itertools
import pickle

# Hyper parameters
WORD_CUT_OFF = 5


def build_vocab(docs, save_path):
  print('Building vocab ...')

  sents = itertools.chain(*[text.split('<sssss>') for text in docs])
  tokenized_sents = [sent.split() for sent in sents]

  # Count the word frequencies
  word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
  print("%d unique words found" % len(word_freq.items()))

  # Cut-off
  retained_words = [w for (w, f) in word_freq.items() if f > WORD_CUT_OFF]
  print("%d words retained" % len(retained_words))

  # Get the most common words and build index_to_word and word_to_index vectors
  # Word index starts from 2, 1 is reserved for UNK, 0 is reserved for padding
  word_to_index = {'PAD': 0, 'UNK': 1}
  for i, w in enumerate(retained_words):
    word_to_index[w] = i + 2
  index_to_word = {i: w for (w, i) in word_to_index.items()}

  print("Vocabulary size = %d" % len(word_to_index))

  with open('{}-w2i.pkl'.format(save_path), 'wb') as f:
    pickle.dump(word_to_index, f)

  with open('{}-i2w.pkl'.format(save_path), 'wb') as f:
    pickle.dump(index_to_word, f)

  return word_to_index


def process_and_save(word_to_index, data, out_file):
  mapped_data = []
  for label, doc in zip(data[4], data[6]):
    mapped_doc = [[word_to_index.get(word, 1) for word in sent.split()] for sent in doc.split('<sssss>')]
    mapped_data.append((label, mapped_doc))

  with open(out_file, 'wb') as f:
    pickle.dump(mapped_data, f)


def read_data(data_file):
  data = pd.read_csv(data_file, sep='\t', header=None, usecols=[4, 6])
  print('{}, shape={}'.format(data_file, data.shape))
  return data


if __name__ == '__main__':
  train_data = read_data('data/yelp-2015-train.txt.ss')
  word_to_index = build_vocab(train_data[6], 'data/yelp-2015')
  process_and_save(word_to_index, train_data, 'data/yelp-2015-train.pkl')

  dev_data = read_data('data/yelp-2015-dev.txt.ss')
  process_and_save(word_to_index, dev_data, 'data/yelp-2015-dev.pkl')

  test_data = read_data('data/yelp-2015-test.txt.ss')
  process_and_save(word_to_index, test_data, 'data/yelp-2015-test.pkl')
