# Import your files here...

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # Get the states and transitions from the file.
    # The states are of the format: dict[ID]: name
    # transitions are of the format: dict[f1]: (f2, f3)
    states, transitions = read_state_or_symbol_file(State_File)
    # Get the symbols and emissions from the file.
    # The symbols are of the format: dict[ID]: name
    # emissions are of the format: dict[f1]: (f2, f3)
    symbols, emissions = read_state_or_symbol_file(Symbol_File)


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...


def read_state_or_symbol_file(file):
    N, i = None, 0
    names = dict()
    frequencies = dict()
    with open(file, 'r') as f:
        for line in f:
            if N is None:
                N = int(line)
            elif N is not None and i < N:
                names[i] = line
                i += 1
            else:
                f1, f2, f3 = map(int, line.split())
                frequencies[f1] = (f2, f3)
    return names, frequencies



if __name__ == '__main__':

    # Question 1
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)