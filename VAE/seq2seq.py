

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np
from utils.distributions import DiagonalGaussian
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib import rnn 

# TODO(ebrevdo): Remove once _linear is fully deprecated.
#linear = rnn_cell._linear  # pylint: disable=protected-access
linear = core_rnn_cell._linear


def prelu(_x):
  with tf.variable_scope("prelu"):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
      initializer=tf.constant_initializer(0.0),
      dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, word_dropout_keep_prob=1, replace_inp=None,
                loop_function=None, scope=None):
  """RNN decoder for the sequence-to-sequence model.

  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    seq_len = len(decoder_inputs)
    #keep = tf.select(tf.random_uniform([seq_len]) < word_dropout_keep_prob,
    keep = tf.where(tf.random_uniform([seq_len]) < word_dropout_keep_prob,
            tf.fill([seq_len], True), tf.fill([seq_len], False))
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          if word_dropout_keep_prob < 1:
            inp = tf.cond(keep[i], lambda: loop_function(prev, i), lambda: replace_inp)
          else:
            inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def beam_rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None,output_projection=None, beam_size=1):
  """RNN decoder for the sequence-to-sequence model.

  Returns:
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    log_beam_probs, beam_path, beam_symbols = [],[],[]
    state_size = int(initial_state.get_shape().with_rank(2)[1])

    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i,log_beam_probs, beam_path, beam_symbols)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      input_size = inp.get_shape().with_rank(2)[1]
      x = inp
      output, state = cell(x, state)

      if loop_function is not None:
        prev = output
      if  i ==0:
          states =[]
          for kk in range(beam_size):
                states.append(state)
          state = tf.reshape(tf.concat(0, states), [-1, state_size])

      outputs.append(tf.argmax(nn_ops.xw_plus_b(
          output, output_projection[0], output_projection[1]), dimension=1))
  return outputs, state, tf.reshape(tf.concat(0, beam_path),[-1,beam_size]), tf.reshape(tf.concat(0, beam_symbols),[-1,beam_size])


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          embedding,
                          num_symbols,
                          embedding_size,
                          word_dropout_keep_prob=1,
                          replace_input=None,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          weight_initializer=None,
                          beam_size=1,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Returns:
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    if not embedding:
      embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size],
              initializer=weight_initializer())

    if beam_size > 1:
        loop_function = _extract_beam_search(
        embedding, beam_size,num_symbols,embedding_size,  output_projection,
        update_embedding_for_previous)
    else:
        loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None

    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    if beam_size > 1:
        return beam_rnn_decoder(emb_inp, initial_state, cell,loop_function=loop_function,
                output_projection=output_projection, beam_size=beam_size)

    return rnn_decoder(emb_inp, initial_state, cell, word_dropout_keep_prob, replace_input,
                       loop_function=loop_function)


def embedding_attention_encoder(encoder_inputs,
                                cell,
                                num_encoder_symbols,
                                embedding_size,
                                dtype=None,
                                scope=None):
  """Embedding sequence-to-sequence model with attention.


  Args:

  Returns:
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_encoder", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    #encoder_cell = rnn_cell.EmbeddingWrapper(
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    return encoder_state, attention_states


def embedding_encoder(encoder_inputs,
                      cell,
                      embedding,
                      num_symbols,
                      embedding_size,
                      bidirectional=False,
                      dtype=None,
                      weight_initializer=None,
                      scope=None):

  with variable_scope.variable_scope(
      scope or "embedding_encoder", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    if not embedding:
      embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size],
              initializer=weight_initializer())
    emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in encoder_inputs]
    if bidirectional:
      _, output_state_fw, output_state_bw = rnn.bidirectional_rnn(cell, cell, emb_inp,
              dtype=dtype)
      encoder_state = tf.concat(1, [output_state_fw, output_state_bw])
    else:
      #_, encoder_state = rnn.rnn(
      _, encoder_state = rnn.static_rnn(
        cell, emb_inp, dtype=dtype)

    return encoder_state


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
  Raises:
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
  Returns:
  Raises:
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses


def autoencoder_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, encoder, decoder, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  Args:
  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        encoder_state = encoder(encoder_inputs[:bucket[0]])
        bucket_outputs, _ = decoder(encoder_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses


def sample(means,
           logvars,
           latent_dim,
           iaf=True,
           kl_min=None,
           anneal=False,
           kl_rate=None,
           dtype=None):
  """Perform sampling and calculate KL divergence.

  Args:
  """
  if iaf:
    with tf.variable_scope('iaf'):
      prior = DiagonalGaussian(tf.zeros_like(means, dtype=dtype),
              tf.zeros_like(logvars, dtype=dtype))
      posterior = DiagonalGaussian(means, logvars)
      z = posterior.sample

      logqs = posterior.logps(z)
      L = tf.get_variable("inverse_cholesky", [latent_dim, latent_dim], dtype=dtype, initializer=tf.zeros_initializer)
      diag_one = tf.ones([latent_dim], dtype=dtype)
      L = tf.matrix_set_diag(L, diag_one)
      mask = np.tril(np.ones([latent_dim,latent_dim]))
      L = L * mask
      latent_vector = tf.matmul(z, L)
      logps = prior.logps(latent_vector)
      kl_cost = logqs - logps
  else:
    noise = tf.random_normal(tf.shape(mean))
    sample = mean + tf.exp(0.5 * logvar) * noise
    kl_cost = -0.5 * (logvars - tf.square(means) -
        tf.exp(logvars) + 1.0)
  kl_ave = tf.reduce_mean(kl_cost, [0]) #mean of kl_cost over batches
  kl_obj = kl_cost = tf.reduce_sum(kl_ave)
  if kl_min:
    kl_obj = tf.reduce_sum(tf.maximum(kl_ave, kl_min))
  if anneal:
    kl_obj = kl_obj * kl_rate

  return latent_vector, kl_obj, kl_cost #both kl_obj and kl_cost are scalar


def encoder_to_latent(encoder_state,
                      embedding_size,
                      latent_dim,
                      num_layers,
                      activation=tf.nn.relu,
                      use_lstm=False,
                      enc_state_bidirectional=False,
                      dtype=None):
  concat_state_size = num_layers * embedding_size
  if enc_state_bidirectional:
    concat_state_size *= 2
  if use_lstm:
    concat_state_size *= 2
    if num_layers > 1:
      encoder_state = list(map(lambda state_tuple: tf.concat(1, state_tuple), encoder_state))
    else:
      encoder_state = tf.concat(1, encoder_state)
  if num_layers > 1:
    encoder_state = tf.concat(1, encoder_state)
  with tf.variable_scope('encoder_to_latent'):
    w = tf.get_variable("w",[concat_state_size, 2 * latent_dim],
      dtype=dtype)
    b = tf.get_variable("b", [2 * latent_dim], dtype=dtype)
    mean_logvar = prelu(tf.matmul(encoder_state, w) + b)
    #mean, logvar = tf.split(1, 2, mean_logvar)
    mean, logvar = tf.split(mean_logvar, 2, 1)

  return mean, logvar


def latent_to_decoder(latent_vector,
                      embedding_size,
                      latent_dim,
                      num_layers,
                      activation=tf.nn.relu,
                      use_lstm=False,
                      dtype=None):

  concat_state_size = num_layers * embedding_size
  if use_lstm:
    concat_state_size *= 2
  with tf.variable_scope('latent_to_decoder'):
    w = tf.get_variable("w",[latent_dim, concat_state_size],
      dtype=dtype)
    b = tf.get_variable("b", [concat_state_size], dtype=dtype)
    decoder_initial_state = prelu(tf.matmul(latent_vector, w) + b)
  if num_layers > 1:
    decoder_initial_state = tuple(tf.split(1, num_layers, decoder_initial_state))
    if use_lstm:
      decoder_initial_state = [tuple(tf.split(1, 2, single_layer_state)) for single_layer_state in decoder_initial_state]
  elif use_lstm:
    decoder_initial_state = tuple(tf.split(1, 2, decoder_initial_state))

  return decoder_initial_state


def variational_autoencoder_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, encoder, decoder, enc_latent, latent_dec, sample, kl_f,
                       probabilistic=False,
                       softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  Raises:
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  KL_divergences = []
  with ops.name_scope(name, "variational_autoencoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        encoder_last_state = encoder(encoder_inputs[:bucket[0]])
        mean, logvar = enc_latent(encoder_last_state)
        if probabilistic:
          latent_vector = sample(mean, logvar)
        else:
          latent_vector = mean
        decoder_initial_state = latent_dec(latent_vector)
        bucket_outputs, _ = decoder(decoder_initial_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        total_size = math_ops.add_n(weights[:bucket[1]])
        total_size += 1e-12
        KL_divergences.append(tf.reduce_mean(kl_f(mean, logvar) / total_size))
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses, KL_divergences


def variational_encoder_with_buckets(encoder_inputs, buckets, encoder,
                       enc_latent, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))

  all_inputs = encoder_inputs
  means = []
  logvars = []
  with ops.name_scope(name, "variational_encoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        encoder_last_state = encoder(encoder_inputs[:bucket[0]])
        mean, logvar = enc_latent(encoder_last_state)
        means.append(mean)
        logvars.append(logvar)

  return means, logvars


def variational_decoder_with_buckets(means, logvars, decoder_inputs,
                       targets, weights,
                       buckets, decoder, latent_dec, sample,
                       softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = decoder_inputs + targets + weights
  losses = []
  outputs = []
  KL_objs = []
  KL_costs = []
  with ops.name_scope(name, "variational_decoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True if j > 0 else None):

        latent_vector, kl_obj, kl_cost = sample(means[j], logvars[j])
        decoder_initial_state = latent_dec(latent_vector)

        bucket_outputs, _ = decoder(decoder_initial_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        total_size = math_ops.add_n(weights[:bucket[1]])
        total_size += 1e-12
        KL_objs.append(tf.reduce_mean(kl_obj / total_size))
        KL_costs.append(tf.reduce_mean(kl_cost / total_size))
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses, KL_objs, KL_costs


def variational_beam_decoder_with_buckets(means, logvars, decoder_inputs,
                       targets, weights,
                       buckets, decoder, latent_dec, kl_f, sample, iaf=False,
                       softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.
  """
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = decoder_inputs + targets + weights
  losses = []
  outputs = []
  beam_paths = []
  beam_path = []
  KL_divergences = []
  with ops.name_scope(name, "variational_decoder_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True if j > 0 else None):
        latent_vector, kl_cost = sample(means[j], logvars[j])
        decoder_initial_state = latent_dec(latent_vector)

        bucket_outputs, _, beam_path, beam_symbol = decoder(decoder_initial_state, decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        beam_paths.append(beam_path)
        beam_symbols.append(beam_symbol)
        total_size = math_ops.add_n(weights[:bucket[1]])
        total_size += 1e-12
        KL_divergences.append(tf.reduce_mean(kl_cost / total_size))
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses, KL_objs, KL_costs
