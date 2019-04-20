# Import your files here...
import re
import numpy as np

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # Get the states and transitions from the file.
    # The states are of the format: dict[ID]: name
    # transitions are of a matrix format with ID as the indices
    states, transitions = read_state_file(State_File)

    N = len(states.keys())
    print('N is {}'.format(N))

    # Get the symbols and emissions from the file.
    # The symbols are of the format: dict[name]: ID
    # emissions are of a matrix format with ID as the indices
    symbols, emissions = read_symbol_file(Symbol_File, N)

    query_tokens = parse_query_file(Query_File)

    print('States are:\n{}'.format(states))
    print('Transitions are:\n{}'.format(transitions))
    print('Symbols are:\n{}'.format(symbols))
    print('Emissions are:\n{}'.format(emissions))
    print('Query tokens are:\n{}'.format(query_tokens))
    tokens_id = []
    # Convert each token into symbol IDs
    for query_token in query_tokens:
        tk = []
        for token in query_token:       
            symbol_id = symbols[token] if token in symbols.keys() else 'UNK'
            tk.append(symbol_id)
        tokens_id.append(tk)
    print('Query tokens id form:', tokens_id)

    # Smoothing the transition probabilities
    transition_probabilities = np.array([[0.0 for _ in range(len(transitions[0]))] for _ in range(len(transitions))])
    for i in range(len(transition_probabilities)):
        for j in range(len(transition_probabilities[0])):
            if states[j] == 'BEGIN':  # ignore when state to transition to is 'BEGIN' since there is no transition to it
                continue
            if states[i] == 'END':  # ignore when state to transition from is 'END' since there is no transition from it
                continue
            transition_probabilities[i, j] = (transitions[i, j] + 1) / (np.sum(transitions[i, :]) + N - 1)
            # transition_probabilities[i, j] = (transitions[i, j]) / (np.sum(transitions[i, :]))
    print('Transition probabilities:\n{}'.format(transition_probabilities))

    # Smoothing the emission probabilities
    M = len(symbols.keys())
    emission_probabilities = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    for i in range(N):
        for j in range(M):
            if states[i] == 'BEGIN' or states[i] == 'END':
                continue
            emission_probabilities[i, j] = (emissions[i, j] + 1) / (np.sum(emissions[i, :]) + M + 1)
    print('Emission probabilities:\n{}'.format(emission_probabilities))


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...


def read_state_file(file):
    N, i = None, 0
    state = dict()
    file = open(file, 'r')
    data = file.read().split('\n')
    file.close()
    frequencies = None
    for line in data:
        if not line:
            continue
        line = line.strip()
        if N is None:
            N = int(line)
        elif N is not None and i < N:
            state[i] = line.strip()
            i += 1
        else:
            if frequencies is None:
                frequencies = [[0 for _ in range(len(state.keys()))] for _ in range(len(state.keys()))]
            f1, f2, f3 = map(int, line.split())
            frequencies[f1][f2] = f3
    return state, np.array(frequencies)


def read_symbol_file(file, N):
    M, i = None, 0
    symbols = dict()
    file = open(file, 'r')
    data = file.read().split('\n')
    file.close()
    frequencies = None
    for line in data:
        if not line:
            continue
        line = line.strip()
        if M is None:
            M = int(line)
        elif M is not None and i < M:
            symbols[line] = i
            i += 1
        else:
            if frequencies is None:
                frequencies = [[0 for _ in range(len(symbols.keys()))] for _ in range(N)]
            f1, f2, f3 = map(int, line.split())
            frequencies[f1][f2] = f3
    return symbols, np.array(frequencies)


def parse_query_file(file):
    delimiters = '*,()/-&'
    tokens = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
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