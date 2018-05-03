
from test.generator import generate_resource
import anomaly_detection

train_dataset = list(generate_resource(100, 'train'))
test_dataset = list(generate_resource(5, 'test')) + [
     '../../../passwd',
     ':(){ :|: & };:',
     'a',
     'a' * 70
]

length_model = anomaly_detection.check_length.train(train_dataset)
distribution_model = anomaly_detection.check_distribution.train(train_dataset)
gramma_model = anomaly_detection.check_gramma.train(train_dataset)

print('train dataset:')
for resource in train_dataset:
    print(f' {resource}')

print('')
print('test dataset:')
for resource in test_dataset:
    print(f' {resource}')
    print(f' - length: {anomaly_detection.check_length.validate(length_model, resource)}')
    print(f' - distribution: {anomaly_detection.check_distribution.validate(distribution_model, resource)}')
    print(f' - gramma: {anomaly_detection.check_gramma.validate(distribution_model, resource)}')
    print('')
