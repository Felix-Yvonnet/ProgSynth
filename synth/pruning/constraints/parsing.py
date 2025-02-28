from dataclasses import dataclass
from typing import (
    Tuple,
    List as TList,
    Type,
    Union,
)
from synth.syntax.grammars.det_grammar import DerivableProgram


from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.program import Variable

# ========================================================================================
# SYMBOLS
# ========================================================================================
SYMBOL_ANYTHING = "_"
SYMBOL_FORBIDDEN = "^"
SYMBOL_SEPARATOR = ","
SYMBOL_SUBTREE = ">"
SYMBOL_AGGREGATOR = "#"
# ========================================================================================
# TOKENS
# ========================================================================================
class Token:
    def __repr__(self) -> str:
        return str(self)


class TokenAnything(Token):
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "Any"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TokenAnything)


@dataclass
class TokenAllow(Token):
    allowed: TList[DerivableProgram]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Allow ({self.allowed})"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TokenAllow) and o.allowed == self.allowed


@dataclass
class TokenForbidSubtree(Token):
    forbidden: TList[DerivableProgram]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"ForbidSubtree ({self.forbidden})"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TokenForbidSubtree) and o.forbidden == self.forbidden


@dataclass
class TokenForceSubtree(Token):
    forced: TList[DerivableProgram]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"ForceSubtree ({self.forced})"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TokenForceSubtree) and o.forced == self.forced


@dataclass
class TokenAtMost(Token):
    to_count: TList[DerivableProgram]
    count: int

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"AtMost {self.count} ({self.to_count})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TokenAtMost)
            and o.to_count == self.to_count
            and o.count == self.count
        )


@dataclass
class TokenAtLeast(Token):
    to_count: TList[DerivableProgram]
    count: int

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"AtLeast {self.count} ({self.to_count})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TokenAtLeast)
            and o.to_count == self.to_count
            and o.count == self.count
        )


@dataclass
class TokenFunction(Token):
    function: TokenAllow
    args: TList[Token]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Func f={self.function} args=({self.args})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TokenFunction)
            and o.function == self.function
            and o.args == self.args
        )


# ========================================================================================
# PARSING
# ========================================================================================
def __next_level__(string: str, start: str, end: str) -> int:
    level = 0
    for i, el in enumerate(string):
        if el == start:
            level += 1
        if el == end:
            level -= 1
            if level == 0:
                return i
    return i


def __parse_next_word__(program: str) -> Tuple[str, int]:
    if program[0] in ["("]:
        end = __next_level__(program, "(", ")")
    else:
        end = program.index(" ") - 1 if " " in program else len(program) - 1
    return program[: end + 1], end + 2


def __str_to_derivable_program__(word: str, grammar: TTCFG) -> TList[DerivableProgram]:
    all_primitives = sorted(
        grammar.primitives_used(), key=lambda p: p.primitive, reverse=True
    )
    if word == SYMBOL_ANYTHING:
        out: TList[DerivableProgram] = all_primitives  # type: ignore
        out += grammar.variables()
        return out
    word = word.strip("(){}")
    allowed = set(
        [word] if not SYMBOL_SEPARATOR in word else word.split(SYMBOL_SEPARATOR)
    )
    primitives: TList[DerivableProgram] = [
        P for P in all_primitives if P.primitive in allowed
    ]
    arg_types = grammar.type_request.arguments()
    for el in allowed:
        if el.startswith("var"):
            varno = int(el[3:])
            primitives.append(Variable(varno, arg_types[varno]))
    return primitives


def __interpret_word__(word: str, grammar: TTCFG) -> Token:
    word = word.strip()
    if word.startswith(SYMBOL_FORBIDDEN):
        forbidden = set(word[1:].split(SYMBOL_SEPARATOR))
        out: TList[DerivableProgram] = [
            P for P in grammar.primitives_used() if P.primitive not in forbidden
        ]
        out += [V for V in grammar.variables() if str(V) not in forbidden]
        return TokenAllow(out)
    elif word.startswith(SYMBOL_SUBTREE):
        cst: Union[
            Type[TokenForceSubtree], Type[TokenForbidSubtree]
        ] = TokenForceSubtree
        str_content = word[1:]
        if str_content.startswith(SYMBOL_FORBIDDEN):
            str_content = str_content[1:]
            cst = TokenForbidSubtree
        return cst(__str_to_derivable_program__(str_content, grammar))
    elif word == SYMBOL_ANYTHING:
        return TokenAnything()
    elif word.startswith(SYMBOL_AGGREGATOR):
        word = word[1:].replace(" ", "")
        end_index = max(word.find("<="), word.find(">="))
        most = word[end_index] == "<"
        considered = word[:end_index]
        content = __str_to_derivable_program__(considered, grammar)
        count = int(word[len(considered) + 2 :])
        if most:
            return TokenAtMost(content, count)
        else:
            return TokenAtLeast(content, count)
    return TokenAllow(__str_to_derivable_program__(word, grammar))


def parse_specification(spec: str, grammar: TTCFG) -> Token:
    spec = spec.replace("\n", "").strip(")(")
    index = 0
    elements = []
    while index < len(spec):
        spec = spec[index:]
        word, index = __parse_next_word__(spec)
        if word.startswith("("):
            token = parse_specification(word, grammar)
        else:
            token = __interpret_word__(word, grammar)
        elements.append(token)
    assert len(elements) > 0
    if isinstance(elements[0], TokenAllow):
        first = elements.pop(0)
        return TokenFunction(first, elements)  # type: ignore
    assert len(elements) == 1
    return elements[0]
