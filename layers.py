import tensorflow as tf
import numpy as np
from utils import get_shape

try:
  from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple



def bidirectional_rnn(cell_fw, cell_bw, inputs, input_lengths,
                      initial_state_fw=None, initial_state_bw=None,
                      scope=None):
  with tf.variable_scope(scope or 'bi_rnn') as scope:
    (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs,
      sequence_length=input_lengths,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=tf.float32,
      scope=scope
    )
    outputs = tf.concat((fw_outputs, bw_outputs), axis=2)

    def concatenate_state(fw_state, bw_state):
      if isinstance(fw_state, LSTMStateTuple):
        state_c = tf.concat(
          (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
        state_h = tf.concat(
          (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
        state = LSTMStateTuple(c=state_c, h=state_h)
        return state
      elif isinstance(fw_state, tf.Tensor):
        state = tf.concat((fw_state, bw_state), 1,
                          name='bidirectional_concat')
        return state
      elif (isinstance(fw_state, tuple) and
            isinstance(bw_state, tuple) and
            len(fw_state) == len(bw_state)):
        # multilayer
        state = tuple(concatenate_state(fw, bw)
                      for fw, bw in zip(fw_state, bw_state))
        return state

      else:
        raise ValueError(
          'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)
    return outputs, state


def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
  score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])
  score_mask_values = score_mask_value * tf.ones_like(scores)
  return tf.where(score_mask, scores, score_mask_values)


def attention(inputs, att_dim, sequence_lengths, scope=None):
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):
    word_att_W = tf.get_variable(name='att_W', shape=[att_dim, 1])

    projection = tf.layers.dense(inputs, att_dim, tf.nn.tanh, name='projection')

    alpha = tf.matmul(tf.reshape(projection, shape=[-1, att_dim]), word_att_W)
    alpha = tf.reshape(alpha, shape=[-1, get_shape(inputs)[1]])
    alpha = masking(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))
    alpha = tf.nn.softmax(alpha)

    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)
    return outputs, alpha
