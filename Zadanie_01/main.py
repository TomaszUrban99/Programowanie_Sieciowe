import numpy as np
import pandas as pd
import random as rd
from Perceptron import Perceptron

def generate_learn_check ( list_learn, list_test, list_input):

    input_count = len(list_input)

    list_learn_count = int(0.8*len(list_input))
    
    in_list = 0

    # Generate random sequence fro list_input
    for i in range(0,5):
        rd.shuffle(list_input)

    while in_list < list_learn_count:
        
        # Random element from input list
        list_learn.append(list_input.pop())
        in_list += 1

    print(in_list)

    for i in list_input:
        list_test.append(list_input.pop())

    

# Read values from file
data = pd.read_csv("iris.data", header=None)

# Read data from Iris Data
y = data.iloc[0:, 4].values

# Empty lists of IRIS types
iris_lists = []

# Find all types of Irises on list
for i in y:
    if i not in iris_lists:
        iris_lists.append(i)

# Iris classificators
iris_classificators = []

X2 = data.iloc[0:,0:].values

list_of_list = []

# Count elements of each class
for i in range(0,len(iris_lists)):
    first_list = []
    for j in X2:
        if j[4] == iris_lists[i]:
            first_list.append(j)
    list_of_list.append(first_list)

list_learn = []
list_check = []

for i in range(0,len(iris_lists)):
    print(len(list_of_list[i]))
    generate_learn_check(list_learn, list_check, list_of_list[i])


# Read data from Iris Data
y1 = []
    
for i in list_learn:
    y1.append(i[4])

X1 = []

for i in list_learn:
    X1.append(i[0:4])


# Import modules for plotting
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Classification for each flower
for i in iris_lists:
    
    y = np.array(y1)
    y = np.where ( y == i, 1, -1)
    
    X = np.array(X1)

    ppn = Perceptron( eta=0.1, epochs=25 )
    
    ppn.train(X,y)

    iris_classificators.append(ppn.w_)

    print(ppn.w_)


# Test input

    
for i in list_check:

    print ( "Klasyfikacja " + str(i[4]))

    max_value = 0
    inp = np.array(i)

    for j in iris_classificators:
        weight = np.array(j)
        cl = np.dot(inp[0:4], weight[1:])
        print(cl)
        
        