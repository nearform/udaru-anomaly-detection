
from nose.tools import *
from _generator import generate_resource

import anomaly_detection.check_distribution as check_distribution


def test_check_distribution():
    model = check_distribution.train(generate_resource(100, 'train'))

    for sequence in generate_resource(5, 'test'):
        assert check_distribution.validate(model, sequence)

    # Check a few invalid samples
    assert not check_distribution.validate(model, '../../../passwd')
    assert not check_distribution.validate(model, ':(){ :|: & };:')
