
import os
import pickle
import os.path as path

from udaru_anomaly_detection \
    import check_length, check_distribution, check_gramma
from udaru_anomaly_detection.tests.generator import generate_resource

dirname = path.dirname(path.realpath(__file__))
modeldir = path.join(dirname, 'models')

train_dataset = list(generate_resource(100, 'train'))
os.makedirs(modeldir, exist_ok=True)

length_model = check_length.train(train_dataset, verbose=True)
with open(path.join(modeldir, 'length-model.pkl'), 'wb') as fp:
    pickle.dump(length_model, fp, protocol=pickle.HIGHEST_PROTOCOL)


distribution_model = check_distribution.train(train_dataset, verbose=True)
with open(path.join(modeldir, 'distribution-model.pkl'), 'wb') as fp:
    pickle.dump(distribution_model, fp, protocol=pickle.HIGHEST_PROTOCOL)


gramma_model = check_gramma.train(train_dataset, verbose=True)
with open(path.join(modeldir, 'gramma-model.pkl'), 'wb') as fp:
    pickle.dump(gramma_model, fp, protocol=pickle.HIGHEST_PROTOCOL)
