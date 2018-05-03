
from nose.tools import *
from _generator import generate_resource

import math

import anomaly_detection.check_gramma as check_gramma

fr3 = 1/3


def test_loss_calculation():
    # Test loss on:
    # ^ -> A -> $
    model = check_gramma.CheckGrammaModel()
    node_a = model.add_node()
    node_a.increment_emission('A')
    model.root.increment_transition(node_a)
    node_a.increment_transition(model.end)

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(3 ** -3))
    assert_almost_equal(model.compute_sequence_log_cost(['A']),
                        math.log(1 * 1 * 1))

    # Test loss on:
    # ^ -> [A] -> $
    node_a.increment_transition(node_a)
    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(3 ** -4))
    assert_almost_equal(model.compute_sequence_log_cost(['A']),
                        math.log(1 * 0.5))
    assert_almost_equal(model.compute_sequence_log_cost(['A', 'A']),
                        math.log(1 * 0.5 * 0.5))

    # Test loss on:
    #  /-> B -\
    # ^ -> [A] -> $
    node_b = model.add_node()
    node_b.increment_emission('B')
    model.root.increment_transition(node_b)
    node_b.increment_transition(model.end)

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(4 ** -7))
    assert_almost_equal(model.compute_sequence_log_cost(['A']),
                        math.log(0.5 * 0.5))
    assert_almost_equal(model.compute_sequence_log_cost(['A', 'A']),
                        math.log(0.5 * 0.5 * 0.5))
    assert_almost_equal(model.compute_sequence_log_cost(['B']),
                        math.log(0.5 * 1 * 1))

    # Test loss on:
    #  /-->   B  --\
    # ^ -> [A = 75% | B = 25%] -> $
    node_a.increment_emission('B')
    node_a.increment_emission('A')
    node_a.increment_emission('A')

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(4 ** -(5 + 5)))
    assert_almost_equal(model.compute_sequence_log_cost(['A']),
                        math.log(0.5 * 0.75 * 0.5))
    assert_almost_equal(model.compute_sequence_log_cost(['A', 'A']),
                        math.log(0.5 * 0.75 * 0.5 * 0.75 * 0.5))
    assert_almost_equal(model.compute_sequence_log_cost(['B']),
                        math.log(0.5 * 1 * 1 + 0.5 * 0.25 * 0.5))
    assert_almost_equal(model.compute_sequence_log_cost(['B', 'B']),
                        math.log(0.5 * 0.25 * 0.5 * 0.25 * 0.5))


def test_merge_node():
    # Build the graph
    #
    #      -A-   -F-
    #     /   \ /   \
    #    /     D     \
    #   /     / \     \
    # ^ +-- B    +- G -+ $
    #   \     \ /     /
    #    \     E     /
    #     \   / \   /
    #      -C-   -H-
    #
    model = check_gramma.CheckGrammaModel()

    def create_node(char):
        node = model.add_node()
        node.increment_emission(char)
        return node

    node_a = create_node('A')
    node_b = create_node('B')
    node_c = create_node('C')
    node_d = create_node('D')
    node_e = create_node('E')
    node_f = create_node('F')
    node_g = create_node('G')
    node_h = create_node('H')

    model.root.increment_transition(node_a)
    model.root.increment_transition(node_b)
    model.root.increment_transition(node_c)

    node_a.increment_transition(node_d)
    node_b.increment_transition(node_d)
    node_b.increment_transition(node_e)
    node_c.increment_transition(node_e)

    node_d.increment_transition(node_f)
    node_d.increment_transition(node_g)
    node_e.increment_transition(node_g)
    node_e.increment_transition(node_h)

    node_f.increment_transition(model.end)
    node_g.increment_transition(model.end)
    node_h.increment_transition(model.end)

    # Check cost of structure
    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(10 ** -(8 + 14)))

    assert_almost_equal(model.compute_sequence_log_cost(['A', 'D', 'F']),
                        math.log(fr3 * 1.0 * 0.5 * 1.0))
    assert_almost_equal(model.compute_sequence_log_cost(['B', 'D', 'G']),
                        math.log(fr3 * 0.5 * 0.5 * 1.0))
    assert_almost_equal(model.compute_sequence_log_cost(['B', 'E', 'G']),
                        math.log(fr3 * 0.5 * 0.5 * 1.0))
    assert_almost_equal(model.compute_sequence_log_cost(['C', 'E', 'H']),
                        math.log(fr3 * 1.0 * 0.5 * 1.0))

    #
    # Merge node E into D
    #
    #       A       F
    #      / \     / \
    #     /   \   /   \
    #    /     \ /     \
    # ^ +- B -[D|E]= G -+ $
    #    \     / \     /
    #     \   /   \   /
    #      \ /     \ /
    #       C       H
    #
    model.merge_node(node_d, node_e)

    # Check cost of structure
    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(9 ** -(10 + 12)))

    assert_almost_equal(model.compute_sequence_log_cost(['A', 'D', 'F']),
                        math.log(fr3 * 1.0 * 0.5 * 0.25 * 1.0))
    assert_almost_equal(model.compute_sequence_log_cost(['B', 'D', 'G']),
                        math.log(fr3 * 1.0 * 0.5 * 0.5 * 1.0))
    assert_almost_equal(model.compute_sequence_log_cost(['B', 'E', 'G']),
                        math.log(fr3 * 1.0 * 0.5 * 0.5 * 1.0))
    assert_almost_equal(model.compute_sequence_log_cost(['C', 'E', 'H']),
                        math.log(fr3 * 1.0 * 0.5 * 0.25 * 1.0))


def test_merge_self_recursive_node():
    def create_node(char):
        node = model.add_node()
        node.increment_emission(char)
        return node

    # Graph:
    # ^ -> [A] -> $
    #  \--> B --/
    model = check_gramma.CheckGrammaModel()
    node_a = create_node('A')
    node_b = create_node('B')

    model.root.increment_transition(node_a)
    model.root.increment_transition(node_b)
    node_a.increment_transition(node_a)
    node_a.increment_transition(model.end)
    node_b.increment_transition(model.end)

    model.merge_node(node_a, node_b)

    assert_almost_equal(node_a.transition, {
        node_a.index: 1*fr3,
        model.end.index: 2*fr3
    })

    # Graph:
    # ^ -> A -> B -> $
    #      \-<-/
    model = check_gramma.CheckGrammaModel()
    node_a = create_node('A')
    node_b = create_node('B')

    model.root.increment_transition(node_a)
    node_a.increment_transition(node_b)
    node_b.increment_transition(node_a)
    node_b.increment_transition(model.end)

    model.merge_node(node_a, node_b)

    assert_almost_equal(node_a.transition, {
        node_a.index: 2*fr3,
        model.end.index: 1*fr3
    })


def test_merge_sequence_self_merge():
    model = check_gramma.CheckGrammaModel()
    dataset = [
        ['A', 'B', 'C', 'A', 'B', 'C', 'D']
    ]

    # Expect
    #
    #       /---<---\
    # ^ -> A -> B -> C -> D -> $
    #
    model.merge_sequence(dataset[0], dataset[0:1])

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(6 ** -10))

    # Test that the graph generalizes to ABC repeating more than the dataset
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'D']
        ),
        math.log(1.0 * 1.0 * 1.0 * 0.5 *
                 1.0 * 1.0 * 1.0 * 0.5 *
                 1.0 * 1.0 * 1.0 * 0.5 *
                 1.0 * 1.0)
    )


def test_merge_sequence():
    model = check_gramma.CheckGrammaModel()
    dataset = [
        ['A', 'A', 'C', 'D'],
        ['B', 'B', 'C', 'D'],
        ['C', 'C', 'C', 'D']
    ]

    # Expect
    # ^ -> [A] --> C -> D -> $
    #
    model.merge_sequence(dataset[0], dataset[0:1])

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(5 ** -(3 + 5)))
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['A', 'A', 'A', 'C', 'D']
        ),
        math.log(1.0 * 0.5 * 0.5 * 0.5 * 1.0 * 1.0)
    )

    # Expect
    # ^ -> [A] -+> C -> D -> $
    #  \-> [B] -/
    model.merge_sequence(dataset[1], dataset[0:2])

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(6 ** -(4 + 8)))
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['A', 'A', 'A', 'C', 'D']
        ),
        math.log(0.5 * 0.5 * 0.5 * 0.5 * 1.0 * 1.0)
    )
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['B', 'B', 'B', 'C', 'D']
        ),
        math.log(0.5 * 0.5 * 0.5 * 0.5 * 1.0 * 1.0)
    )

    # Expect
    #  /-> [A] -\
    # ^ --------+> [C] -> D -> $
    #  \-> [B] -/
    model.merge_sequence(dataset[2], dataset[0:3])

    assert_almost_equal(model.compute_prior_log_cost(),
                        math.log(6 ** -(4 + 10)))
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['A', 'A', 'A', 'C', 'D']
        ),
        math.log(fr3 * 0.5 * 0.5 * 0.5 * 1.0 * 0.6 * 1.0)
    )
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['B', 'B', 'B', 'C', 'D']
        ),
        math.log(fr3 * 0.5 * 0.5 * 0.5 * 1.0 * 0.6 * 1.0)
    )
    assert_almost_equal(
        model.compute_sequence_log_cost(
            ['C', 'C', 'C', 'C', 'D']
        ),
        math.log(fr3 * 0.4 * 0.4 * 0.4 * 0.6 * 1.0)
    )
