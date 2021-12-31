from typing import Dict, Set, Tuple

from synth.pruning.pruner import Pruner
from synth.syntax.program import Program
from synth.syntax.type_system import Arrow, Type


class UseAllVariablesPruner(Pruner[Tuple[Type, Program]]):
    def __init__(self) -> None:
        super().__init__()
        self._cached_variables_set: Dict[Type, Set[int]] = {}

    def __get_var_set__(self, treq: Type) -> Set[int]:
        if treq not in self._cached_variables_set:
            if isinstance(treq, Arrow):
                self._cached_variables_set[treq] = set(range(len(treq.arguments())))
            else:
                self._cached_variables_set[treq] = set()

        return self._cached_variables_set[treq]

    def accept(self, obj: Tuple[Type, Program]) -> bool:
        treq, prog = obj
        target = self.__get_var_set__(treq)
        return prog.used_variables() == target
