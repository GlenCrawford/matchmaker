import argparse

def parse_arguments():
  parser = argparse.ArgumentParser(
    allow_abbrev = False,
    description = 'K-Nearest Neighbors machine learning model to find the best matches within a set of OkCupid profiles.'
  )

  parser.add_argument(
    '--matches',
    action = 'store',
    default = 40, # Just because that's what fits on my screen ;)
    type = int,
    dest = 'matches_to_retrieve',
    help = 'The number of matching profiles to find (default: 40).'
  )

  parser.add_argument(
    '--force-training',
    action = 'store_true',
    dest = 'force_training',
    help = 'Train the model even if a previously trained and saved model can be loaded and used (default: false).'
  )

  return parser.parse_args()
