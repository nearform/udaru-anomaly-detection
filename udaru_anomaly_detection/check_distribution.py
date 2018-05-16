
import typing
import collections
import itertools

import scipy.stats
import numpy as np

"""
Checks if the distribution of characters is unlikely.

The expected distribution of charecters is calculated from the training
dataset. Some of the charecters are grouped together, this is to increase
generality and the degrees of freedom in the statistical test.

This charecter grouping shouldn't compromise the models ability to detect
anomalies. Attacks are usually done using special charecters, such as / and ..
in filepaths and not using standard latin letters and numbers.

The statistical test, is a chi-squared test. This test depends on the
chi-squared distribution and is poorly implemented in JavaScript. Those
implementation may work for some input combinations, but don't expect them
to work for all. This may prove a problem in usecases like this, where the
parameter input isn't something that can be controlled.

For a better implementation, the scipy python library is used here. This
in turns depends on the "special function" part of the Cephes library. This
library could be ported to JavaScript if needed.

Cephes Library: https://github.com/scipy/scipy/tree/master/scipy/special/cephes
"""

collaps_chars = dict()
collaps_chars.update({char: '<0-9>' for char in '0123456789'})
collaps_chars.update({char: '<a-f>' for char in 'abcdef'})
collaps_chars.update({char: '<A-F>' for char in 'ABCDEF'})
collaps_chars.update({char: '<g-z>' for char in 'ghijklmnopqurtuvwxyz'})
collaps_chars.update({char: '<G-Z>' for char in 'GHIJKLMNOPQURTUVWXYZ'})


class CheckDistributionModel(typing.NamedTuple):
    order: typing.Dict[str, int]
    frequency: np.ndarray


def compute_unordered_frequency(sequence: str) -> \
        typing.DefaultDict[str, float]:
    # Count the number of occurrences for each charecters. Note that some
    # charecters are collaped into a single symbol.
    counts = collections.Counter(
        collaps_chars[char] if char in collaps_chars else char
        for char in sequence
    )

    # Convery counts to frequency
    frequency = collections.defaultdict(float)
    frequency.update({
        char: count / len(sequence) for char, count in counts.items()
    })

    return frequency


def train(sequences: typing.List[str], verbose: bool=False) \
        -> CheckDistributionModel:
    if verbose:
        print(f'Training DistributionModel with {len(sequences)} sequences')

    # Compute frequency for each sequence
    frequencies = [
        compute_unordered_frequency(sequence) for sequence in sequences
    ]

    # Get a list of all charecters
    all_chars = set(itertools.chain.from_iterable((
        frequency.keys() for frequency in frequencies
    )))

    # compute the frequency over all sequences
    global_frequency = {
        char: np.mean([frequency[char] for frequency in frequencies])
        for char in all_chars
    }
    global_frequency_sorted = sorted(global_frequency.items(),
                                     key=lambda item: -item[1])

    # Return a model containing:
    #  - order: a mapping between charecter and index
    #  - frequency: the expected frequency in the \chi^2 test
    return CheckDistributionModel(
        order={
            char: i
            for i, (char, frequency)
            in enumerate(global_frequency_sorted)
        },
        frequency=np.fromiter((
            frequency for char, frequency in global_frequency_sorted
        ), dtype=np.float64)
    )


def validate(model: CheckDistributionModel, sequence: str,
             threshold: float=0.1) -> bool:
    # Construct ordered frequency array
    unordered_frequency = compute_unordered_frequency(sequence)
    ordered_frequency = np.zeros_like(model.frequency)

    for char, frequency in unordered_frequency.items():
        if char in model.order:
            index = model.order[char]
            ordered_frequency[index] = frequency
        else:
            # Invalid charecter
            return False

    # Compare input sequence with reference frequency
    (chisq, p) = scipy.stats.chisquare(ordered_frequency, model.frequency)

    # Threshold the p-value
    return p >= threshold
