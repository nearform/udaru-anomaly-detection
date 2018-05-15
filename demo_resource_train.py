
import os
import pickle

from _generator import generate_resource
import anomaly_detection

train_dataset = list(generate_resource(100, 'train'))
os.makedirs('models', exist_ok=True)

length_model = anomaly_detection.check_length.train(train_dataset,
                                                    verbose=True)
with open('models/length-model.pkl', 'wb') as fp:
    pickle.dump(length_model, fp, protocol=pickle.HIGHEST_PROTOCOL)

distribution_model = anomaly_detection.check_distribution.train(train_dataset,

                                                                verbose=True)
with open('models/distribution-model.pkl', 'wb') as fp:
    pickle.dump(distribution_model, fp, protocol=pickle.HIGHEST_PROTOCOL)


gramma_model = anomaly_detection.check_gramma.train(train_dataset,
                                                    verbose=True)
with open('models/gramma-model.pkl', 'wb') as fp:
    pickle.dump(gramma_model, fp, protocol=pickle.HIGHEST_PROTOCOL)
