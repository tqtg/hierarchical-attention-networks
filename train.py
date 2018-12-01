import tensorflow as tf
from datetime import datetime
from data_reader import DataReader
from model import Model
from utils import read_vocab, count_parameters, load_glove

# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", 'checkpoints',
                       """Path to checkpoint folder""")
tf.flags.DEFINE_string("log_dir", 'logs',
                       """Path to log folder""")

tf.flags.DEFINE_integer("cell_dim", 50,
                        """Hidden dimensions of GRU cells (default: 50)""")
tf.flags.DEFINE_integer("att_dim", 100,
                        """Dimensionality of attention spaces (default: 100)""")
tf.flags.DEFINE_integer("emb_size", 200,
                        """Dimensionality of word embedding (default: 200)""")
tf.flags.DEFINE_integer("num_classes", 5,
                        """Number of classes (default: 5)""")

tf.flags.DEFINE_integer("num_checkpoints", 1,
                        """Number of checkpoints to store (default: 1)""")
tf.flags.DEFINE_integer("num_epochs", 20,
                        """Number of training epochs (default: 20)""")
tf.flags.DEFINE_integer("batch_size", 64,
                        """Batch size (default: 64)""")
tf.flags.DEFINE_integer("display_step", 20,
                        """Number of steps to display log into TensorBoard (default: 20)""")

tf.flags.DEFINE_float("learning_rate", 0.0005,
                      """Learning rate (default: 0.0005)""")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      """Maximum value of the global norm of the gradients for clipping (default: 5.0)""")
tf.flags.DEFINE_float("dropout_rate", 0.5,
                      """Probability of dropping neurons (default: 0.5)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

if not tf.gfile.Exists(FLAGS.checkpoint_dir):
  tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

if not tf.gfile.Exists(FLAGS.log_dir):
  tf.gfile.MakeDirs(FLAGS.log_dir)

train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')


def loss_fn(labels, logits):
  onehot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
  cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                       logits=logits)
  tf.summary.scalar('loss', cross_entropy_loss)
  return cross_entropy_loss


def train_fn(loss):
  trained_vars = tf.trainable_variables()
  count_parameters(trained_vars)

  # Gradient clipping
  gradients = tf.gradients(loss, trained_vars)

  clipped_grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
  tf.summary.scalar('global_grad_norm', global_norm)

  # Add gradients and vars to summary
  # for gradient, var in list(zip(clipped_grads, trained_vars)):
  #   if 'attention' in var.name:
  #     tf.summary.histogram(var.name + '/gradient', gradient)
  #     tf.summary.histogram(var.name, var)

  # Define optimizer
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
  train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars),
                                       name='train_op',
                                       global_step=global_step)
  return train_op, global_step


def eval_fn(labels, logits):
  predictions = tf.argmax(logits, axis=-1)
  correct_preds = tf.equal(predictions, tf.cast(labels, tf.int64))
  batch_acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
  tf.summary.scalar('accuracy', batch_acc)

  total_acc, acc_update = tf.metrics.accuracy(labels, predictions, name='metrics/acc')
  metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  metrics_init = tf.variables_initializer(var_list=metrics_vars)

  return batch_acc, total_acc, acc_update, metrics_init


def main(_):
  vocab = read_vocab('data/yelp-2015-w2i.pkl')
  glove_embs = load_glove('glove.6B.{}d.txt'.format(FLAGS.emb_size), FLAGS.emb_size, vocab)
  data_reader = DataReader(train_file='data/yelp-2015-train.pkl',
                           dev_file='data/yelp-2015-dev.pkl',
                           test_file='data/yelp-2015-test.pkl')

  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    model = Model(cell_dim=FLAGS.cell_dim,
                  att_dim=FLAGS.att_dim,
                  vocab_size=len(vocab),
                  emb_size=FLAGS.emb_size,
                  num_classes=FLAGS.num_classes,
                  dropout_rate=FLAGS.dropout_rate,
                  pretrained_embs=glove_embs)

    loss = loss_fn(model.labels, model.logits)
    train_op, global_step = train_fn(loss)
    batch_acc, total_acc, acc_update, metrics_init = eval_fn(model.labels, model.logits)
    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    train_writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

    print('\n{}> Start training'.format(datetime.now()))

    epoch = 0
    valid_step = 0
    test_step = 0
    train_test_prop = len(data_reader.train_data) / len(data_reader.test_data)
    test_batch_size = int(FLAGS.batch_size / train_test_prop)
    best_acc = float('-inf')

    while epoch < FLAGS.num_epochs:
      epoch += 1
      print('\n{}> Epoch: {}'.format(datetime.now(), epoch))

      sess.run(metrics_init)
      for batch_docs, batch_labels in data_reader.read_train_set(FLAGS.batch_size, shuffle=True):
        _step, _, _loss, _acc, _ = sess.run([global_step, train_op, loss, batch_acc, acc_update],
                                         feed_dict=model.get_feed_dict(batch_docs, batch_labels, training=True))
        if _step % FLAGS.display_step == 0:
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          train_writer.add_summary(_summary, global_step=_step)
      print('Training accuracy = {:.2f}'.format(sess.run(total_acc) * 100))

      sess.run(metrics_init)
      for batch_docs, batch_labels in data_reader.read_valid_set(test_batch_size):
        _loss, _acc, _  = sess.run([loss, batch_acc, acc_update], feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        valid_step += 1
        if valid_step % FLAGS.display_step == 0:
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          valid_writer.add_summary(_summary, global_step=valid_step)
      print('Validation accuracy = {:.2f}'.format(sess.run(total_acc) * 100))

      sess.run(metrics_init)
      for batch_docs, batch_labels in data_reader.read_test_set(test_batch_size):
        _loss, _acc, _  = sess.run([loss, batch_acc, acc_update], feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        test_step += 1
        if test_step % FLAGS.display_step == 0:
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          test_writer.add_summary(_summary, global_step=test_step)
      test_acc = sess.run(total_acc) * 100
      print('Testing accuracy = {:.2f}'.format(test_acc))

      if test_acc > best_acc:
        best_acc = test_acc
        # saver.save(sess, FLAGS.checkpoint_dir)
      print('Best testing accuracy = {:.2f}'.format(test_acc))

  print("{} Optimization Finished!".format(datetime.now()))
  print('Best testing accuracy = {:.2f}'.format(best_acc * 100))


if __name__ == '__main__':
  tf.app.run()
