# Import your files here...
import re

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # Get the states and transitions from the file.
    # The states are of the format: dict[ID]: name
    # transitions are of the format: dict[f1]: (f2, f3)
    states, transitions = read_state_file(State_File)
    # Get the symbols and emissions from the file.
    # The symbols are of the format: dict[name]: ID
    # emissions are of the format: dict[f1]: (f2, f3)
    symbols, emissions = read_symbol_file(Symbol_File)

    tokens = parse_query_file(Query_File)




# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...


def read_state_file(file):
    N, i = None, 0
    state = dict()
    frequencies = dict()
    with open(file, 'r') as f:
        for line in f:
            if N is None:
                N = int(line)
            elif N is not None and i < N:
                state[i] = line
                i += 1
            else:
                f1, f2, f3 = map(int, line.split())
                frequencies[f1] = (f2, f3)
    return state, frequencies


def read_symbol_file(file):
    N, i = None, 0
    symbols = dict()
    frequencies = dict()
    with open(file, 'r') as f:
        for line in f:
            if N is None:
                N = int(line)
            elif N is not None and i < N:
                symbols[line] = i
                i += 1
            else:
                f1, f2, f3 = map(int, line.split())
                frequencies[f1] = (f2, f3)
    return symbols, frequencies


def parse_query_file(file):
    delimiters = '*,()/-&'
    tokens = []
    with open(file, 'r') as f:
        for line in f:
            token = []
            queries = line.split()
            for q in queries:
                tk = re.split(r'(\*|\,|\(|\)|\/|-|\&)', q)
                for t in tk:
                    if t != '':
                        token.append(t)
            tokens.append(token)
    return tokens


if __name__ == '__main__':

    # Question 1
    State_File ='./toy_example/State_File'
    Symbol_File='./toy_example/Symbol_File'
    Query_File ='./toy_example/Query_File'
    viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)