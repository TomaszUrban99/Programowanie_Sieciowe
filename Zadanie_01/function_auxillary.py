import random as rd

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

    for i in range(0,len(list_input)):
        list_test.append(list_input.pop())

