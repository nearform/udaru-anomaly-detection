
import pickle
import os.path as path

from udaru_anomaly_detection \
    import check_length, check_distribution, check_gramma
from udaru_anomaly_detection.tests.generator import generate_resource

dirname = path.dirname(path.realpath(__file__))
modeldir = path.join(dirname, 'models')

train_dataset = list(generate_resource(100, 'test'))
test_dataset = list(generate_resource(5, 'test')) + [
     '../../../passwd',
     ':(){ :|: & };:',
     'a',
     'a' * 70,
     'res::ricky:/sl/jennifersaunders',
     'res:/sl/:ricky:/jennifersaunders'
]


with open(path.join(modeldir, 'length-model.pkl'), 'rb') as fp:
    length_model = pickle.load(fp)

with open(path.join(modeldir, 'distribution-model.pkl'), 'rb') as fp:
    distribution_model = pickle.load(fp)

with open(path.join(modeldir, 'gramma-model.pkl'), 'rb') as fp:
    gramma_model = pickle.load(fp)

print('train dataset:')
for resource in train_dataset:
    print(f' {resource}')

print('')
print('test dataset:')
for resource in test_dataset:
    print(f' {resource}')
    print(f' - length: {check_length.validate(length_model, resource)}')
    print(f' - distribution: {check_distribution.validate(distribution_model, resource)}')
    print(f' - gramma: {check_gramma.validate(gramma_model, resource)}')
    print('')
