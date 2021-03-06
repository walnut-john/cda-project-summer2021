import pickle
import xgboost as xgb
import tensorflow as tf

# Sci-Kit Learn
def save_scikit_model(model, filename='mlp_model.sav'):
    pickle.dump(model, open(filename, 'wb'))

def load_scikit_model(filename='mlp_model.sav'):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


# XGBoost
def save_xgb_model(model, filename='xgb_model.json'):
    model.save_model(filename)

def load_xgb_model(filename='xgb_model.json'):
    model2 = xgb.XGBClassifier()
    model2.load_model(filename)
    return model2


# TensFlow
def save_tf_model(model, filename='tf_model.h5'):
    model.save(filename)

def load_tf_model(filename='tf_model.h5'):
    new_model = tf.keras.models.load_model(filename)
    return new_model
