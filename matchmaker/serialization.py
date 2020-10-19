import os.path
import joblib

# Relative from the project root directory.
NEAREST_NEIGHBORS_MODEL_PATH = 'models/nearest_neighbors_entire_population.skmodel'

def load_model():
  if os.path.isfile(NEAREST_NEIGHBORS_MODEL_PATH):
    return joblib.load(NEAREST_NEIGHBORS_MODEL_PATH) 
  else:
    return None

def save_model(nearest_neighbors_model):
  joblib.dump(nearest_neighbors_model, NEAREST_NEIGHBORS_MODEL_PATH)
