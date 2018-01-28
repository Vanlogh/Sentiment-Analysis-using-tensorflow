import tensorflow as tf 


# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
	cleaners='english_cleaners',

	#Preprocessing/feeder
	max_len = 200,

	#Model
	num_classes = 2,
	embedding_dim = 256,

	#Training
	batch_size = 64,
	reg_weight = 10e-6,
	learning_rate = 10e-3,
	adam_beta1 = 0.9,
	adam_beta2 = 0.999,
	adam_epsilon = 10e-6,

	)

def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)