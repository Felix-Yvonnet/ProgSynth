import json
from typing import Any, Callable, Dict, Tuple, List as TList

import tqdm

from synth.task import Task, Dataset
from synth.specification import PBE, Example
from synth.syntax.type_system import INT, FunctionType, List, Type
from synth.syntax.program import Function, Primitive, Program, Variable

from deepcoder import dsl

name2type = {p.primitive: p.type for p in dsl.list_primitives}


def __convert__(load: Callable[[], Dataset[PBE]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = len([0 for t in tasks if t.solution])
    print(
        f"Converted {len(tasks)} tasks {int(100 * sols / len(tasks))}% containing solutions"
    )


def convert_deepcoder(
    file: str = "deepcoder_dataset/T=3_train.json",
    output_file: str = "deepcoder.pickle",
) -> None:
    def load() -> Dataset[PBE]:
        tasks: TList[Task[PBE]] = []
        with open(file, "r") as fd:
            raw_tasks: TList[Dict[str, Any]] = json.load(fd)
            for raw_task in tqdm.tqdm(raw_tasks, desc="converting"):
                name: str = raw_task["program"]
                raw_examples: TList[Dict[str, Any]] = raw_task["examples"]
                inputs = [raw_example["inputs"] for raw_example in raw_examples]
                outputs: TList = [raw_example["output"] for raw_example in raw_examples]

                prog, type_request = __deepcoder_str2prog(name)
                examples = [Example(inp, out) for inp, out in zip(inputs, outputs)]
                tasks.append(
                    Task[PBE](type_request, PBE(examples), prog, {"name": name})
                )
        return Dataset(tasks, metadata={"dataset": "deepcoder", "source:": file})

    __convert__(load, output_file)


def __deepcoder_str2prog(s: str) -> Tuple[Program, Type]:
    parts = s.split("|")
    stack: TList[Program] = []
    var: int = 0
    type_stack: TList[Type] = []
    for part in parts:
        subparts = part.split(",")
        name = subparts.pop(0)
        if name == "LIST":
            stack.append(Variable(var, List(INT)))
            var += 1
            type_stack.append(List(INT))
            continue
        if name == "INT":
            stack.append(Variable(var, INT))
            var += 1
            type_stack.append(INT)
            continue
        if name not in name2type.keys():
            name = name + "[" + subparts.pop(0) + "]"
        primitive = Primitive(name, name2type[name])
        targets = [int(x) for x in subparts]
        arguments = [stack[x] for x in targets]
        stack.append(Function(primitive, arguments))
    type_stack.append(stack[-1].type)
    type_request = FunctionType(*type_stack)
    return stack[-1], type_request


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert deepcoder original dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "deepcoder.pickle",
    }

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON deepcoder file to be converted",
    )
    argument_parser.add_argument(
        "-o",
        "--output",
        type=str,
        action="store",
        default=argument_default_values["output"],
        help=f"Output dataset file in ProgSynth format (default: '{argument_default_values['output']}')",
    )
    parsed_parameters = argument_parser.parse_args()
    convert_deepcoder(parsed_parameters.file, parsed_parameters.output)
