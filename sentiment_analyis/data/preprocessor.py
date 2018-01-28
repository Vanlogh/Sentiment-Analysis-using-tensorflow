import pandas as pd 
import os
import argparse
from utils.text import text_to_sequence


def read_data(file, training):
	if training:
		data = pd.read_csv(file, delimiter='\t', names=['label', 'text'])
	else:
		data = pd.read_csv(file, names=['text'])
	
	return data

def create_vocabulary(texts, vocab_file):
	word_vocab = dict()
	def func(text):
		global word_vocab
		for word in text.split():
			if word not in vocab:
				word_vocab[word] = 1
			else:
				word_vocab[word] += 1
		return 0

	_ = texts.apply(func)

	with open(vocab_file, 'w') as file:
		for key, value in sorted(word_vocab.iteritems(), key=lambda k, v: (v, k)):
			file.write('{}\t{}'.format(value, key))

	print('created vocabulary at {}'.vocab_file)

def preprocess(in_file, out_file, training):
	data = read_data(in_file, training=training)
	texts = data['text']
	create_vocabulary(texts, out_file)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default=os.path.dirname(os.path.realpath(__file__)))
	parser.add_argument('--input', default='train.csv')
	parser.add_argument('--output_dir', default='../datasets')
	parser.add_argument('--training', type=bool, default=True, help='Set this to False to preprocess text set')
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	output_dir = args.output_dir

	if args.training:
		input_file = 'train.csv'
		output_file = os.path.join(output_dir, 'train.csv')
	else:
		input_file = 'test.csv'
		output_file = of.path.join(output_dir, 'test.csv')

	preprocess(input_file, output_file)


if __name__ == '__main__':
	main()