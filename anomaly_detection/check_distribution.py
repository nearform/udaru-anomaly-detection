
import typing
import collections
import itertools

import scipy.stats
import numpy as np

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


def train(sequences: typing.List[str]):
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
