# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #ex1
    model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
    cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.97],
                                                              [0.01, 0.03]],
                       evidence=['C'],
                       evidence_card=[2])
    cpd_a = TabularCPD(variable='A', variable_card=2,
                       values=[[0.9999, 0.05, 0.98, 0.02],
                               [0.0001, 0.95, 0.02, 0.98]],
                       evidence=['C', 'I'],
                       evidence_card=[2, 2])
    model.add_cpds(cpd_c, cpd_i, cpd_a)
    model.check_model()

    #ex2
    infer = VariableElimination(model)
    result = infer.query(variables=['C'], evidence={'A': 1})
    print(result)

    #ex3
    infer = VariableElimination(model)
    result = infer.query(variables=['I'], evidence={'A': 0})
    print(result)

    pos = nx.circular_layout(model)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
    plt.show()
