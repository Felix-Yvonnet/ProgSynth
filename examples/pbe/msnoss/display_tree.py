import pickle
import graphviz

from synth.syntax.dsl import DSL
from examples.pbe.dsl_loader import load_DSL
from synth.syntax import CFG, enumerate_prob_u_grammar, UCFG, ProbUGrammar
from synth.pruning.constraints import add_dfta_constraints
from synth.syntax.type_system import Arrow, INT, List
from typing import List as UList
from synth.syntax import (
    CFG,
    Function,
    Lambda,
)
from tqdm import tqdm
# sudo apt-get install graphviz

from mcts import MCTS, Node

import argparse

parser = argparse.ArgumentParser(description="Display the ")
parser.add_argument(
    "-ns",
    "--no-show",
    action="store_true",
    default=False,
    help="To avoid displaying the result",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="mcts_output.png",
    help="output file (default: 'mcts_output.png')",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="verbose mode",
)
parser.add_argument(
    "-f",
    "--filename",
    type=str,
    default="",
    help="The place of the file to transform",
)
parser.add_argument(
    "--dsl",
    type=str,
    default="deepcoder",
    help="The dsl",
)
parser.add_argument(
    "-t",
    "--tests",
    action="store_true",
    default=False,
    help="to test on random data",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=57,
    help="The seed to use for random",
)
parser.add_argument(
    "--format",
    type=str,
    default="pdf",
    help="The place of the file to transform",
)

parameters = parser.parse_args()
no_show: bool = parameters.no_show
output: str = parameters.output
verbose: bool = parameters.verbose
file: str = parameters.filename
test: bool = parameters.tests
seed: int = parameters.seed
dsl_name: str = parameters.dsl
format: str = parameters.format


assert test or file, "You should provide some object to display"



module = load_DSL(dsl_name)
dsl: DSL = module.dsl

if test:
    cfg = CFG.depth_constraint(dsl, Arrow(INT, INT), 3)
    dfta = add_dfta_constraints(cfg, [], "(+ (- _))", True)
    ucfg = UCFG.from_DFTA_with_ngrams(dfta, 2)

    import random
    import sys

    random.seed(seed)
    mcts = MCTS(5)
    for program in tqdm(enumerate_prob_u_grammar(ProbUGrammar.uniform(ucfg))):
        sys.stdout.write('\rfound program ' + str(program))
        mcts += (program, random.randint(0, 5))
        sys.stdout.flush()
    list_mcts = [mcts]
    if verbose: print("data successfully generated")

else:
    thing = pickle.load(file)
    if isinstance(thing, UList[MCTS]):
        list_mcts = thing
    elif isinstance(thing, MCTS):
        list_mcts = [thing]
    else:
        raise TypeError("Unknown type, expected MCTS or list of MCTS")
    if verbose: print("data successfully loaded from", file)


"""
expected formatting of file:
a set of list of tuples (complete_function_name, success) the list are separated by new lines and the sets by semicolon with the total number of tests at the beguinning
For example:
`filename.csv`
```
10
(DROP (SUM (SCAN1L[max] (SCAN1L[min] var0))) var0),7
(MAP[/2] (ZIPWITH[max] var0 (MAP[*2] (REVERSE var0)))),9
(MAP[-1] var0),3
;3
(MAP[*-1] (FILTER[>0] (REVERSE var0))),1
(ZIPWITH[*] var0 (MAP[*4] (ZIPWITH[+] (MAP[*4] var0) (SCAN1L[+] var0)))),3
```

That way splitting between give us a list of all tasks and then splitting between new lines give us the different tests with their success rate (knowing the total test as the first element)
"""


if verbose: print("The final tree has: ", len(list_mcts[0]), " nodes")


def no_ext(file: str) -> str:
    splitted = file.split(".")
    if len(splitted) <= 1:
        return file
    else:
        return ".".join(splitted)

def get_directory(file: str) -> str:
    splitted = file.split("/")
    if len(splitted) <= 1:
        return file
    else:
        return "/".join(splitted)

class Pointer: 
    """ Just to have a pointer to be incremented """
    def __init__(self, x: int):
        self.x = x

def to_dot(dot: graphviz.Digraph, prog: Node, p: Pointer) -> int:
    if isinstance(prog.program, Function):
        name = "function call"
    elif isinstance(prog.program, Lambda):
        name = "lambda"
    else:
        # of type Constant, Variable or Primitive
        name = str(prog.program)

    dot.node(f"{p.x}", f"name = {name}, value = {prog.value}")
    curr_node = p.x
    p.x += 1

    if isinstance(prog.program, Function):
        functions = p.x
        dot.node(f"{p.x}", "functions:")
        dot.edge(f"{curr_node}", f"{p.x}")
        p.x += 1
        assert prog.children[0] is not None
        for poss in prog.children[0].values():
            new_num = to_dot(dot, poss, p)
            dot.edge(f"{functions}", f"{new_num}")

        args = p.x
        dot.node(f"{p.x}", "arguments:")
        dot.edge(f"{curr_node}", f"{p.x}")
        p.x += 1

        for i in range(1, len(prog.children)):
            child = prog.children[i]
            tmp_node = p.x
            dot.node(f"{p.x}", f"possibilities for arg {i}:")
            dot.edge(f"{args}", f"{p.x}")
            p.x += 1
            assert child is not None
            for poss in child.values():
                new_num = to_dot(dot, poss, p)
                dot.edge(f"{tmp_node}", f"{new_num}")

    else:
        for child in prog.children:
            tmp_node = p.x
            dot.node(f"{p.x}", "possibilities:")
            dot.edge(f"{curr_node}", f"{p.x}")
            p.x += 1
            assert child is not None
            for poss in child.values():
                new_num = to_dot(dot, poss, p)
                dot.edge(f"{tmp_node}", f"{new_num}")
    return curr_node


def to_dots(mcts: MCTS) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        f"{no_ext(output)}",
        comment=f"Plotting of the Monte Carlo Tree Search related to the file {file}",
        format=format,
    )
    dot.node("root", "#")
    p = Pointer(0)
    for node in mcts.children.values():
        new_num = to_dot(dot, node, p)
        dot.edge("root", f"{new_num}")
    return dot

for mcts in list_mcts:
    dot = to_dots(mcts)
    if verbose: print("mcts successfully transformed into dot")

    out_file = dot.render(directory=output, view=not no_show)
    if verbose: print(f"file saved at {out_file}")


"""

nr_vertices = 25

G = Graph()
G.add_vertices(range(7))
G.add_edges([(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)])


nr_vertices = 7
v_label = list(map(str, range(nr_vertices)))

#G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
lay = G.layout('rt')

position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(G) # sequence of edges
E = [e.tuple for e in G.es] # list of edges

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2*M-position[k][1] for k in range(L)]
Xe = []
Ye = []
for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

labels = v_label

## Create Plotly Traces
import plotly
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   )) # edges
fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  mode='markers',
                  name='bla',
                  marker=dict(symbol='circle-dot',
                                size=18,
                                color='#6175c1',    #'#DB4551', # inside
                                line=dict(color='rgb(50,50,50)', width=1) # boundary
                                ),
                  text=labels,
                  hoverinfo='text',
                  opacity=0.8
                  )) # nodes




def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    assert len(text)==L, 'The lists pos and text must have the same len'
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=text[k],
                x=pos[k][0], y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations


## sep1
axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

fig.update_layout(title= 'Tree with Reingold-Tilford Layout',
              annotations=make_annotations(position, v_label),
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'import graphviz
              )
fig.show(config={'scrollZoom': True})

"""
