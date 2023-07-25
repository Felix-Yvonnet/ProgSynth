from synth.syntax import (
    Program,
    Function,
    Lambda,
    ProbDetGrammar,
    ProbUGrammar,
)

from synth.syntax.type_system import Type

from synth.syntax.grammars.grammar import DerivableProgram

from typing import Dict, List, Optional, Tuple, Union
from math import exp


def local_eq(p1: Program, p2: Program) -> bool:
    """returns True if the program are locally equal ie if they should be on the same node in the mcts"""
    if isinstance(p1, Function) and isinstance(p2, Function):
        return True
    if isinstance(p2, Lambda) and isinstance(p1, Lambda):
        return True
    if type(p1) == type(p2):
        # type Constant, Variable or Primitive
        return p1 == p2
    return False


def is_local_key_in(
    program: Program, possibilities: Optional[Dict[Program, "Node"]]
) -> Tuple[bool, Optional[Program]]:
    if possibilities is None:
        return False, None
    for key in possibilities.keys():
        if local_eq(program, key):
            return True, key
    return False, None


class Node(object):
    """Represents a node of the MCTS.
    
    Args:
        program (Program): the first element of this program represents the element represented in that node.
        value (float): the value of the node: ie the evaluation of the program.
    """
    def __init__(self, program: Program, value: float) -> None:
        self.type: Type = program.type
        self._value: float = value

        # For counting the number of occurences of this call
        self._seen = 1
        
        self.children: List[Optional[Dict[Program, "Node"]]] = []
        self.possibilities: Dict[Program, List["Node"]] = {}

        if isinstance(program, Function):

            # For a function call the first child represents the function and the others represent the arguments
            function_node = Node(program.function, value)
            self.children.append({program.function: function_node})

            # And now we add the arguments
            for input in program.arguments:
                self.children.append({input: Node(input, value)})

        elif isinstance(program, Lambda):
            self.children.append({program.body: Node(program.body, value)})

    @property
    def value(self) -> float:
        return self._value / self._seen

    def __iadd__(self, other: Tuple[Program, float]) -> "Node":
        """add the program "other" that starts the same as "self" to the tree and propagate the result up the tree.

        Args:
            other (Program, float): a program that we want to add to the tree and the evaluation of this program (should be a proportion of the success made)

        Returns:
            Node: the node updated with the possibility to read the program other
        """

        # just a bit of renaming: other is now the program and we extracted the value
        another, value = other

        self.update_value(value)

        if isinstance(another, Function):
            if len(self.children) < len(another.arguments) + 1:
                self.children += [None] * (
                    len(another.arguments) + 1 - len(self.children)
                )
            for i, (old_inputs, new_input) in enumerate(
                zip(self.children, [another.function] + another.arguments)
            ):
                is_in, key = is_local_key_in(new_input, old_inputs)
                if not is_in:
                    # We have a new node that didn't exist before so we have to create it
                    new_node = Node(new_input, value)
                    if old_inputs is not None:
                        old_inputs[new_input] = new_node
                    else:
                        self.children[i] = {new_input: new_node}
                else:
                    # The node already exists, we can just skip to the next node and see what happen next
                    old_inputs[key] += (new_input, value)  # type: ignore

        if isinstance(another, Lambda):
            # Quite the same for lambda: it is just a passing point for the body so we just transere th info
            is_in, key = is_local_key_in(another.body, self.children[0])
            if not is_in and key is not None:
                new_node = Node(another.body, value)
                self.children[0][key] = new_node  # type: ignore
            else:
                self.children[0][another.body] += (another.body, value) # type: ignore

        else:
            # the two programs ends the same: with a constant, a variable or a primitive so we do nothing
            pass

        return self

    def update_value(self, value: float) -> None:
        """self.value should return the mean of the values seen througn this node"""
        self._value += value
        self._seen += 1

    def operation(self, values: List[float], test_num: int) -> float:
        """
        take a list of values corresponding to the evaluation of the inputs of a function and return the expected value of the composition by this function.
        The exponential part helps giving more importance to results that matches a lot of tests.
        Maybe useless but still useful :)
        """
        return sum([exp(value) - 1 for value in values]) / (
            ((1 - exp(test_num + 1)) / (1 - exp(1)) - test_num - 1) * len(values)
        )

    def __len__(self) -> int:
        if isinstance(self.program, Lambda) or isinstance(self.program, Function):
            return 1 + sum(
                sum(len(poss) for poss in child.values()) for child in self.children if child is not None
            )
        else:
            return 1


class MCTS(object):
    """The root of the Monte Carlo Tree Search. 
    It hides all methods and operations done with the Node-objects and give a simpler interface.

    Args:
        total_tests: the total number of tests done for the specific task for which the MCTS is used.
    """
    
    def __init__(self, total_tests: int) -> None:
        self.total_tests = total_tests
        self.children: dict = dict()

    def __iadd__(self, other: Tuple[Program, int]) -> "MCTS":
        """add a new program to the Monte Carlo Tree Search

        Args:
            other (Tuple[Node, int]): a program to add and its valuation

        Returns:
            MCTS: the newly incremented mcts with the program
        """
        another, value = other
        is_in, key = is_local_key_in(another, self.children)
        if is_in:
            self.children[key] += (another, value)
        else:
            self.children[another] = Node(another, value)
        return self

    def __len__(self) -> int:
        return sum(len(child) for child in self.children.values())

    def grammar_embedding(self, pcfgs: Union[List[ProbDetGrammar], List[ProbUGrammar]]) -> None:
        assert False, "TODO"
