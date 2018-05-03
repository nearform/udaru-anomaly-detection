
import copy
import math
import typing
import queue

collaps_chars = dict()
collaps_chars.update({char: '<0-9>' for char in '0123456789'})
collaps_chars.update({char: '<a-f>' for char in 'abcdef'})
collaps_chars.update({char: '<A-F>' for char in 'ABCDEF'})
collaps_chars.update({char: '<g-z>' for char in 'ghijklmnopqurtuvwxyz'})
collaps_chars.update({char: '<G-Z>' for char in 'GHIJKLMNOPQURTUVWXYZ'})

GrammaNodeT = typing.TypeVar('GrammaNode', bound='GrammaNode')
KeyType = typing.TypeVar('Key')
ValueType = typing.TypeVar('Value')
TokensType = typing.List[str]


def tokenize(sequence: str) -> TokensType:
    return [
        collaps_chars[char] if char in collaps_chars else char
        for char in sequence
    ]


class SparseDefaultDict(typing.DefaultDict[KeyType, ValueType]):
    def __missing__(self, key: KeyType) -> ValueType:
        return self.default_factory()


class SequenceSolution(typing.NamedTuple):
    log_properbility: float
    path: typing.List[int]


class GrammaNode:
    index: int
    is_root: bool
    is_end: bool
    allow_merge: bool
    emission: SparseDefaultDict[str, float]
    num_aggregated_emissions: int
    transition: SparseDefaultDict[int, float]
    num_aggregated_transitions: int
    parents: typing.Set[int]

    def __init__(self, index: int, is_root: bool=False, is_end: bool=False):
        self.index = index
        self.is_root = is_root
        self.is_end = is_end
        self.allow_merge = False
        self.emission = SparseDefaultDict(float)
        self.num_aggregated_emissions = 0
        self.transition = SparseDefaultDict(float)
        self.num_aggregated_transitions = 0
        self.parents = set()

    def stringify_emission(self) -> str:
        if self.is_root:
            return '^'
        elif self.is_end:
            return '$'
        else:
            return ' | '.join([
                f'{char} = {properbility}'
                for char, properbility in self.emission.items()
            ])

    def stringify_transition(self) -> str:
            return ', '.join([
                f'{properbility} -> {index}'
                for index, properbility in self.transition.items()
            ])

    def stringify(self) -> str:
        return (f'{self.index}: ({self.stringify_emission()}) ==> '
                f'{{{self.stringify_transition()}}}')

    def increment_emission(self, inc_char: str):
        n = self.num_aggregated_emissions
        rescale = n / (n + 1)

        # rescale all existing emissions to make room for the incremented char
        for char, properbility in self.emission.items():
            self.emission[char] = rescale * properbility

        # Insert or increment the `inc_char`
        self.emission[inc_char] += 1 / (n + 1)
        self.num_aggregated_emissions = n + 1

    def increment_transition(self, inc_destination: GrammaNodeT):
        inc_index = inc_destination.index
        n = self.num_aggregated_transitions
        rescale = n / (n + 1)

        # rescale all existing emissions to make room for the incremented char
        for index, properbility in self.transition.items():
            self.transition[index] = rescale * properbility

        # Insert or increment the `inc_char`
        self.transition[inc_index] += 1 / (n + 1)
        self.num_aggregated_transitions = n + 1

        # Add this node as parent to the description
        inc_destination.parents.add(self.index)

    def merge_emissions(self, node: GrammaNodeT):
        # Merge emissions
        total_aggregated_emissions = (self.num_aggregated_emissions +
                                      node.num_aggregated_emissions)
        self_emission_rescale = (self.num_aggregated_emissions /
                                 total_aggregated_emissions)
        node_emission_rescale = (node.num_aggregated_emissions /
                                 total_aggregated_emissions)

        for char in (self.emission.keys() | node.emission.keys()):
            self.emission[char] = (
                self_emission_rescale * self.emission[char] +
                node_emission_rescale * node.emission[char]
            )

        self.num_aggregated_emissions = total_aggregated_emissions

    def merge_transitions(self, node: GrammaNodeT):
        # Merge outgoing transitions
        total_aggregated_transitions = (self.num_aggregated_transitions +
                                        node.num_aggregated_transitions)
        self_transition_rescale = (self.num_aggregated_transitions /
                                   total_aggregated_transitions)
        node_transition_rescale = (node.num_aggregated_transitions /
                                   total_aggregated_transitions)

        # Exclude transitions that will become a self-recursive transition
        # when merged. They need to be treated specially.
        non_self_recursive_transitions = (
            (self.transition.keys() | node.transition.keys()) -
            {self.index, node.index}
        )
        for index in non_self_recursive_transitions:
            self.transition[index] = (
                self_transition_rescale * self.transition[index] +
                node_transition_rescale * node.transition[index]
            )

        # For the self-recursive transition, there are a few more links to
        # consider when merging.
        self_recursive_link = (
            # The self-recursive link in `self`
            self_transition_rescale * self.transition[self.index] +
            # Link from `self` to `node` becomes self-recursive
            self_transition_rescale * self.transition[node.index] +
            # The self-recursive link in `node` is transfered to `self`
            node_transition_rescale * node.transition[node.index] +
            # Link from `node` to `self` becomes self-recursive
            node_transition_rescale * node.transition[self.index]
        )
        # Only set the self-recursive link, if a self-recursive link was
        # produced
        if self_recursive_link > 0.0:
            self.transition[self.index] = self_recursive_link
        # Also remove transition to the `node` as this has now become
        # a self-recursive link
        if node.index in self.transition:
            del self.transition[node.index]

        self.num_aggregated_transitions = total_aggregated_transitions


class CheckGrammaModel:
    root: GrammaNode
    end: GrammaNode
    nodes: typing.Dict[int, GrammaNode]
    index_counter: int

    def __init__(self):
        self.root = GrammaNode(0, is_root=True)
        self.end = GrammaNode(1, is_end=True)
        self.nodes = dict()
        self.nodes[self.root.index] = self.root
        self.nodes[self.end.index] = self.end
        self.index_counter = 2

    def copy(self):
        return copy.deepcopy(self)

    def stringify(self) -> str:
        output = ''
        for node in self.nodes.values():
            output += node.stringify() + '\n'
        return output

    def compute_prior_log_cost(self) -> float:
        # Compute prior loss as:
        #   \prod_{s \in States} N^(-e_s) * N^(-t_s)
        # All those multiplications are not nummerically stable, so compute
        # the cost in log space.
        #   \sum_{s \in States} log(N) * (-e_s) + log(N) * (-t_s)
        #   \sum_{s \in States} log(N) * (-t_s - e_s)
        #   log(N) \sum_{s \in States} -(t_s + e_s)
        #   -log(N) \sum_{s \in States} (t_s + e_s)
        # Note, that the root and end nodes are not treated a emitting nodes.
        # However, the start node does have transitions.
        connections = sum(
            3 * (len(node.emission) - 1) + 1 + len(node.transition)
            for node in self.nodes.values()
            if not node.is_root and not node.is_end
        )
        connections += len(self.root.transition)

        return - math.log(len(self.nodes)) * connections

    def sequence_solutions(self, tokenized_sequence: TokensType) -> \
            typing.List[SequenceSolution]:
        q = queue.Queue()
        q.put((0, 0.0, self.root, [self.root.index]))

        sequence_length = len(tokenized_sequence)
        solutions = []

        # Find all possible paths for `tokenized_sequence` and compute the
        # log properbility of that path. Store the result in `solutions`.
        while not q.empty():
            (sequence_index, log_p, node, path) = q.get()
            next_char = (None if sequence_index == sequence_length else
                         tokenized_sequence[sequence_index])

            # Check each transition
            for transition, transition_p in node.transition.items():
                next_node = self.nodes[transition]

                # If there is no next char, the next_node must be the end-node
                if next_char is None:
                    if next_node.is_end:
                        solutions.append(SequenceSolution(
                            log_properbility=log_p + math.log(transition_p),
                            path=path + [next_node.index]
                        ))
                # The transition points to a node with matching emission
                elif next_char in next_node.emission:
                    q.put((
                        sequence_index + 1,
                        log_p + math.log(transition_p) +
                              + math.log(next_node.emission[next_char]),
                        next_node,
                        path + [next_node.index]
                    ))

        return solutions

    def sequence_properbility(self, tokenized_sequence: TokensType) \
            -> float:
        solutions = self.sequence_solutions(tokenized_sequence)

        if len(solutions) == 0:
            return 0

        # P(D|M) = sum_{p \in paths} P(D|M,p)
        return sum(map(
            lambda solution: math.exp(solution.log_properbility),
            solutions
        ))

    def compute_sequence_log_cost(self, tokenized_sequence: TokensType) \
            -> float:
        solutions = self.sequence_solutions(tokenized_sequence)

        if len(solutions) == 0:
            raise RuntimeError(f'no solutions found for {tokenized_sequence}')

        # P(D|M) = sum_{p \in paths} P(D|M,p)
        # log(P(D|M)) = log( sum_{p \in paths} exp(log(P(D|M,p))) )
        return math.log(sum(map(
            lambda solution: math.exp(solution.log_properbility),
            solutions
        )))

    def compute_log_cost(self, dataset: typing.List[TokensType]) \
            -> float:
        # unnormalized_posterior = p(data) * p(prior)
        # log(unnormalized_posterior) = log(p(data)) + log(p(prior))
        log_prior = self.compute_prior_log_cost()
        log_data = sum(
            map(self.compute_sequence_log_cost, dataset)
        )
        unnormalized_log_posterior = log_data + log_prior
        return -unnormalized_log_posterior

    def add_node(self) -> GrammaNode:
        new_node = GrammaNode(self.index_counter)
        self.index_counter += 1
        self.nodes[new_node.index] = new_node
        return new_node

    def merge_node(self, keep_node: GrammaNode, remove_node: GrammaNode):
        keep_node.merge_emissions(remove_node)
        keep_node.merge_transitions(remove_node)

        # Merge the ingoing transitions
        for remove_parent_index in remove_node.parents:
            # Skip the self-recursive link
            if remove_parent_index == keep_node.index:
                continue

            parent_node = self.nodes[remove_parent_index]

            # Transfer the properbility (flux) from the `remove_node` to the
            # `keep_node`.
            parent_node.transition[keep_node.index] += \
                parent_node.transition[remove_node.index]
            del parent_node.transition[remove_node.index]

            # Now that the parent_node from the remove_node is a parent of
            # keep_node, update the parents set.
            keep_node.parents.add(remove_parent_index)

        # Redirect the `parent` of the removed node children.
        for child_index in remove_node.transition.keys():
            # Skip the self-recursive link
            if child_index == remove_node.index:
                continue

            child_node = self.nodes[child_index]
            child_node.parents.remove(remove_node.index)
            child_node.parents.add(keep_node.index)

        # Finally remove the node from the graph table
        del self.nodes[remove_node.index]

    def add_unmerged_sequence(self, tokenized_sequence: TokensType) \
            -> typing.List[GrammaNode]:
        unmerged_nodes = []

        # Build a path between root and end node, that contains the new
        # sequence.
        previuse_node = self.root
        previuse_char = None
        for char in tokenized_sequence:
            # It always pays off on the baysian prior to merge subsequent nodes
            # with the same emission. The only penalty is on posterior
            # properbility is P(X|M), however the penalty here is usually
            # very small. Because this dramatically decrease the number of
            # nodes, improves generalization, and is almost always correct
            # identical subsequent nodes are merged, without computing the
            # posterior properbility:
            #
            # ^ -> A -> A -> A -> / -> B -> B -> B -> $
            #
            # Should be merged to
            #
            # ^ ->  A  -> / ->  B  -> $
            #      ^ |         ^ |
            #      \_/         \_/
            if char == previuse_char:
                # Merge if emission is identical
                previuse_node.increment_transition(previuse_node)
                previuse_node.increment_emission(char)
            else:
                # Create new node if emission is *not* identical
                new_node = self.add_node()
                unmerged_nodes.append(new_node)
                new_node.increment_emission(char)
                previuse_node.increment_transition(new_node)
                previuse_node = new_node
        previuse_node.increment_transition(self.end)

        return unmerged_nodes

    def find_optimal_merge(self, node: GrammaNode,
                           dataset: typing.List[TokensType]):
        best_model_cost = self.compute_log_cost(dataset)
        best_merge_node = None

        for suggested_merge_node in self.nodes.values():
            # Nodes that doesn't allow for merge are either the root or end
            # node, or a node futher down the unmerged sequence that is
            # currently being merged.
            if not suggested_merge_node.allow_merge:
                continue

            # Create a copy of the current model, such that the cost of
            # a potential merge can be computed without interfering the main
            # model.
            suggested_model = self.copy()
            suggested_model.merge_node(
                suggested_model.nodes[suggested_merge_node.index],
                suggested_model.nodes[node.index]
            )
            suggested_model_cost = suggested_model.compute_log_cost(dataset)

            if suggested_model_cost < best_model_cost:
                best_model_cost = suggested_model_cost
                best_merge_node = suggested_merge_node

        return best_merge_node

    def merge_sequence(self, tokenized_sequence: TokensType,
                       dataset: typing.List[TokensType]):
        # Fast path
        solutions = self.sequence_solutions(tokenized_sequence)
        if len(solutions) > 0:
            # If tokenized_sequence is allready representable by the network.
            # If so, a more complex network won't benfit because of P(M) and
            # the focus should be on P(D|M) that can likely be optimized by
            # just increment the highest properbility path.
            best_solution = solutions[0]
            for solution in solutions[1:]:
                if solution.log_properbility > best_solution.log_properbility:
                    best_solution = solution

            # increment transition and emission along the path
            for prev_node_index, this_node_index, char in zip(
                best_solution.path[:-2],
                best_solution.path[1:-1],
                tokenized_sequence
            ):
                prev_node = self.nodes[prev_node_index]
                this_node = self.nodes[this_node_index]

                prev_node.increment_transition(this_node)
                this_node.increment_emission(char)

            # In the above the end node is excluded, so increment the
            # transition between the last real node `this_node` and the end
            # node.
            self.nodes[best_solution.path[-2]].increment_transition(self.end)

        else:
            # Slow path
            nodes = self.add_unmerged_sequence(tokenized_sequence)
            for node in nodes:
                best_merge_node = self.find_optimal_merge(node, dataset)
                if best_merge_node is None:
                    # If there are no optimial merges, then mark this node
                    # as done. This allows futures unmerged nodes to merge into
                    # this node.
                    node.allow_merge = True
                else:
                    # Found an optimal merge, execute merge on the master model
                    # and remove the new node from the list.
                    self.merge_node(best_merge_node, node)


def train(sequences: TokensType) -> CheckGrammaModel:
    model = CheckGrammaModel()

    # Add remaining sequences by merge
    tokenized_sequences = []
    for i, sequence in enumerate(sequences):
        print(f'{i}: {sequence}')
        tokenized_sequence = tokenize(sequence)
        tokenized_sequences.append(tokenized_sequence)
        model.merge_sequence(tokenized_sequence, tokenized_sequences)

    return model


def validate(model: CheckGrammaModel, sequence: str,
             threshold: float=0.0) -> bool:
    return model.sequence_properbility(tokenize(sequence)) > threshold
