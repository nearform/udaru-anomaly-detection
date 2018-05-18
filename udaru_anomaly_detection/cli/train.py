
import datetime
import os
import os.path as path
import pickle

from udaru_anomaly_detection \
    import check_length, check_distribution, check_gramma
from udaru_anomaly_detection.trail.read import trail_read


def train(args):
    reader = trail_read(
        to_date=datetime.datetime.strptime(getattr(args, 'to'), "%Y-%m-%d"),
        from_date=datetime.datetime.strptime(getattr(args, 'from'), "%Y-%m-%d")
    )
    train_dataset = list(map(lambda item: item['subject']['id'], reader))

    os.makedirs(args.modeldir, exist_ok=True)

    length_model = check_length.train(train_dataset, verbose=True)
    with open(path.join(args.modeldir, 'length-model.pkl'), 'wb') as fp:
        pickle.dump(length_model, fp, protocol=pickle.HIGHEST_PROTOCOL)

    distribution_model = check_distribution.train(train_dataset, verbose=True)
    with open(path.join(args.modeldir, 'distribution-model.pkl'), 'wb') as fp:
        pickle.dump(distribution_model, fp, protocol=pickle.HIGHEST_PROTOCOL)

    gramma_model = check_gramma.train(train_dataset, verbose=True)
    with open(path.join(args.modeldir, 'gramma-model.pkl'), 'wb') as fp:
        pickle.dump(gramma_model, fp, protocol=pickle.HIGHEST_PROTOCOL)
