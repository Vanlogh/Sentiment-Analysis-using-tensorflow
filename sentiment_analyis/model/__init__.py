from .Analyser import Analyser


def create_model(name, hparams):
  if name == 'Analyser':
    return Analyser(hparams)
  else:
    raise Exception('Unknown model: ' + name)