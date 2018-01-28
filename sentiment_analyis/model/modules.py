import tensorflow as tf 
from hparams import hparams
from tensorflow.contrib.rnn import LSTMStateTuple
from .zoneout_LSTM import ZoneoutLSTMCell

def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	drop_rate = 0.5

	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched)
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training,
								 name='dropout_{}'.format(scope))

def enc_conv_layers(inputs, is_training, kernel_size=(2, ), channels=512, activation=tf.nn.relu, scope=None):
	if scope is None:
		scope = 'enc_conv_layers'

	with tf.variable_scope(scope):
		x = inputs
		for i in range(3):
			x = conv1d(x, kernel_size, channels, activation,
							is_training, 'conv_layer_{}_'.format(i + 1)+scope)
	return x

def bidirectional_LSTM(inputs, scope, is_training):
	with tf.variable_scope(scope):
		outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
												ZoneoutLSTMCell(256,
																is_training,
																zoneout_factor_cell=0.1,
																zoneout_factor_output=0.1,),
												ZoneoutLSTMCell(256, 
																is_training,
																zoneout_factor_cell=0.1,
																zoneout_factor_output=0.1,),
												inputs,
												dtype=tf.float32)

		#Concatenate c states and h states from forward
		#and backward cells
		encoder_final_state_c = tf.concat(
			(fw_state.c, bw_state.c), 1)
		encoder_final_state_h = tf.concat(
			(fw_state.h, bw_state.h), 1)

		#Get the final state to pass as initial state to decoder
		final_state = LSTMStateTuple(
			c=encoder_final_state_c,
			h=encoder_final_state_h)

	return tf.concat(outputs, axis=2), final_state # Concat forward + backward outputs and final states

def projection(x, is_training, shape=512, activation=None, scope=None):
	drop_rate = 0.5

	if scope is None:
		scope = 'linear_projection'

	with tf.variable_scope(scope):
		# if activation==None, this returns a simple linear projection
		# else the projection will be passed through an activation function
		output = tf.contrib.layers.fully_connected(x, shape, activation_fn=activation, 
												   biases_initializer=tf.zeros_initializer(),
												   scope=scope)
		return tf.layers.dropout(output, rate=drop_rate, training=is_training,
									name='dropout_{}'.format(scope))

def projection_layers(x, shape=[512, 512], activation=tf.nn.relu, scope=None):
	if scope is None:
		scope = 'projection_layer'

	with tf.variable_scope(scope):
		n_layers = len(shape)
		for i in range(n_layers):
			x = projection(x, shape=shape[i], activation=activation, scope='{}_{}'.format(scope, i+1))
	return x

def logit_layer(x, logits_dim, scope=None):
	if scope is None:
		scope = 'logit_layer'

	with tf.variable_scope(scope):
		output = projection(x, is_training=False, shape=logits_dim)

	return output