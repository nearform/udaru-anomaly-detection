
from nose.tools import *
from .generator import generate_resource

import udaru_anomaly_detection.check_length as check_length


def test_check_length():
    model = check_length.train(generate_resource(100, 'train'))

    for sequence in generate_resource(5, 'test'):
        assert check_length.validate(model, sequence)

    # Check a few invalid samples
    assert not check_length.validate(model, 'a')
    assert not check_length.validate(model, 'a' * 1000)

    # Check valid range
    valid_range = []
    for i in range(0, 100):
        valid = check_length.validate(model, '.' * i)

        if len(valid_range) == 0 and valid:
            valid_range.append(i)
        elif len(valid_range) == 1 and not valid:
            valid_range.append(i)
        elif len(valid_range) == 2:
            break

    assert_equal(valid_range, [4, 50])
