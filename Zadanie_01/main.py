import numpy as np
import pandas as pd
import random as rd
from Perceptron import Perceptron
from Adaline import Adaline    
from function_auxillary import generate_learn_check

# Read values from file
data = pd.read_csv("iris.data", header=None)

# Read data from Iris Data (iris.data)
y = data.iloc[0:, 4].values
X2 = data.iloc[0:,0:].values

# Empty lists of IRIS types
iris_lists = []

# Find all types of Irises on list
for i in y:
    if i not in iris_lists:
        iris_lists.append(i)

iris_lists.pop()

list_of_list = []

# Split input data
# based on Iris types
for i in range(0,len(iris_lists)):
    first_list = []
    for j in X2:
        if j[4] == iris_lists[i]:
            first_list.append(j)
    list_of_list.append(first_list)

# Lists for storing elements for learning and checking
list_learn = []
list_check = []

# Generate learn and check sets
for i in range(0,len(iris_lists)):
    print(len(list_of_list[i]))
    generate_learn_check(list_learn, list_check, list_of_list[i])

# Transform lists

y1 = []
for i in list_learn:
    y1.append(i[4])

X1 = []
for i in list_learn:
    X1.append(i[0:4])


# Import modules for plotting
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Lists for storing classificators
iris_classificators_perceptron = []
iris_classificators_adaline = []

# Classification for each flower
for i in iris_lists:
    
    print(i)

    y = np.array(y1)
    y = np.where ( y == i, 1, -1)
    
    print(y)

    X = np.array(X1)

    # PERCEPTRON
    ppn = Perceptron( eta=0.0001, epochs=1000 ) 
    ppn.train(X,y)
    iris_classificators_perceptron.append(ppn)

    # ADALINE
    ada = Adaline ( eta = 0.0001, epochs=1000 )
    ada.train(X,y)
    iris_classificators_adaline.append(ada)

    # Print weight vector for each method
    print ("Perceptron:")
    print(ppn.w_)
    print ("Adaline:")
    print(ada.w_)


for i in list_check:

  print ( "Klasyfikacja " + str(i[4]))

  inp = np.array(i)

  print ( "Perceptron: ")
  
  for j in iris_classificators_perceptron:
      cl = np.dot(inp[0:4], j.w_[1:]) + j.w_[0]
      print(cl)

  print ( "Adaline: ")
      
  for j in iris_classificators_adaline:
      cl = np.dot(inp[0:4], j.w_[1:]) + j.w_[0]
      print(cl)
      