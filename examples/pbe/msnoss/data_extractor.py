from mcts import MCTS


def condition(data, i) :
    return True
    
def extract_data_from_format():
    with open("/home/felix/ENS/l3/Stage/ProgSynth/examples/pbe/machin.csv","r") as fd:
        """expected formatting:
        ;total_tests:probability,evaluation
        probability,evaluation
        ...
        ;total_tests:probability,evaluation
        probability,evaluation
        ...
        
        
        output:
        ["all tasks"
            ["a single task"
                total_tests,
                ["evaluation of programs"
                    probability,
                    evaluation
                ]
            ]
        ]
        """
        # some black magic to get it in the right format
        return list(map(lambda x: [int(x.split(":")[0]), list(map(lambda y: (float(y.split(",")[0]), int(y.split(",")[1])), x.split(":")[1].split("\n")[:-1]))], fd.read()[1:].split(";")))


def to_pcfg(old_pcfg, explored_programs, total_tests):
    mcts = MCTS(total_tests)
    for program in explored_programs: mcts += program
    """
    on parcourt l'arbre de cfg et on regarde le futur : on a n noeuds visités dans mcts et m possibilités dans la grammaire alors on attribue aux non visités une proba 1 / m et aux autres une proba valeur_du_noeud / n on visite ceux qu'on a pas vu et ceux qui ont bien marchésavec proba >>
    
    """


