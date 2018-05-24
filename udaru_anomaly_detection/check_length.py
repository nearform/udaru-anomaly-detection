
import typing

import numpy as np

"""
Checks if the length is unlikely.

Given that the distribution of sequence length is unknown and it is
unlikely any resonable assumtions about it can be made, the Chebyshev
inequality is used instead.

The Chebyshev inequality says that the properbility that something (x)
deviates more from the mean than a threshold (t), is less than `\sigma^2 / t^2`
                    p(|x - \mu| > t) < \sigma^2 / t^2

This is reformulated to "the properbility that something (x)
deviates more from the mean than the current deviation (|l - \mu|). Where `l`
is the current resource string length.
              p(|x - \mu| > |l - \mu|) < \sigma^2 / (l - \mu)^2
"""


class CheckLengthModel(typing.NamedTuple):
    mean: float
    var: float


def train(sequences: typing.Iterable[str], verbose: bool=False) \
        -> CheckLengthModel:
    if verbose:
        print(f'Training LengthModel with {len(sequences)} sequences')

    lengths = np.fromiter(map(len, sequences), int)

    return CheckLengthModel(
        mean=np.mean(lengths),
        var=np.var(lengths)
    )


def validate(model: CheckLengthModel, sequence: str,
             threshold: float=0.1) -> bool:
    upper_bound = model.var / (len(sequence) - model.mean) ** 2

    return upper_bound >= threshold
