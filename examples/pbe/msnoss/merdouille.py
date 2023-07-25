from time import sleep
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import matplotlib.animation as animation





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
    #data = list(map(lambda x: [int(x.split(":")[0]), list(map(lambda y: (float(y.split(",")[0]), int(y.split(",")[1])), x.split(":")[1].split("\n")[:-1]))], fd.read()[1:].split(";")))
    y = fd.readline()
    y = y[1:].split(":")
    total = int(y[0])
    probas = []
    results = []
    y = y[1]
    while y and y[0] == ';':
        prob, rez = y.split(",")
        probas.append(float(prob))
        results.append(int(rez))
        y = fd.readline()
    
    
    
    




"""
with open("examples/pbe/msnoss/test_dataset_model_heap_search_base.csv", "r") as fd:
    x = []
    y = fd.readline()
    y = fd.readline()
    while y:
        x+=[y.split(",")[4]]
        y = fd.readline()

    


task = data[0]
total, program_results = task
print(total)
plt.subplot(211)
plt.plot(range(100), [program_result[0] for program_result in program_results[:100]])
plt.subplot(212)
plt.plot(range(len(program_results)), [program_result[1] for program_result in program_results])

print(len([program_result[1] for program_result in program_results if program_result[1]> 0]), len(program_results))

plt.show()
        """


fig, ax = plt.subplots()
ln, = ax.plot([], [], 'r')


def update(frame):
     plt.clf()
     plt.plot(range(len(probas[frame])), probas[frame])
     sleep(1.3)
     return ln,

ani = animation.FuncAnimation(fig, update, frames=len(probas), repeat = True)
plt.show()


