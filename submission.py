# Import your files here...
import re
import numpy as np
import math

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # Get the states and transitions from the file.
    states, transitions = read_state_file(State_File)

    # find BEGIN and END id
    begin_id, end_id = None, None
    for id in states.keys():
        if states[id] == 'BEGIN':
            begin_id = id
        if states[id] == 'END':
            end_id = id
    N = len(states.keys())

    # Get the symbols and emissions from the file. 
    symbols, emissions = read_symbol_file(Symbol_File, N)
    query_tokens = parse_query_file(Query_File)
    tokens_id = []
    # Convert each token into symbol IDs
    for query_token in query_tokens:
        tk = []
        for token in query_token:       
            symbol_id = symbols[token] if token in symbols.keys() else len(symbols.keys())  # Give UNK the last id
            tk.append(symbol_id)
        tokens_id.append(tk)

    # Smoothing the transition probabilities
    transition_probabilities = np.array([[0.0 for _ in range(len(transitions[0]))] for _ in range(len(transitions))])
    for i in range(len(transition_probabilities)):
        for j in range(len(transition_probabilities[0])):
            if states[j] == 'BEGIN':  # ignore when state to transition to is 'BEGIN' since there is no transition to it
                continue
            if states[i] == 'END':  # ignore when state to transition from is 'END' since there is no transition from it
                continue
            if states[i] == 'BEGIN' and states[j] == 'END':  # cannot go from begin to end straight away?
                continue
            transition_probabilities[i, j] = (transitions[i, j] + 1) / (np.sum(transitions[i, :]) + N - 1)

    # Smoothing the emission probabilities
    M = len(symbols.keys())+1 # +1 for UNK
    emission_probabilities = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    for i in range(N):
        for j in range(M):
            if states[i] == 'BEGIN' or states[i] == 'END':
                continue
            emission_probabilities[i, j] = (emissions[i, j] + 1) / (np.sum(transitions[i, :]) + M + 1)

    # Process each query
    for query in tokens_id:
        # setup dp
        T1 = np.array([[0.0 for _ in range(len(query)+2)] for _ in range(N)])
        T2 = np.array([[0.0 for _ in range(len(query)+2)] for _ in range(N)])

        # starting transition probabilities
        T1[:, 0] = transition_probabilities[begin_id, :]
        T2[:, 0] = begin_id
        prev = 0

        for i in range(1, len(query)+1):
            obs = query[i-1]
            for state in states.keys():
                T1[state, i], T2[state, i] = max([(T1[k, prev] * 
                                                   transition_probabilities[k, state] * 
                                                   emission_probabilities[state, obs], k) for k in states.keys()])
            prev = i
        # transite to END?
        for state in states.keys():
            T1[state, -1], T2[state, -1] = max([(T1[k, prev] * 
                                               transition_probabilities[k, state], k) for k in states.keys()])
        print(T1)
        print(T2)
        # backtract to get path?
        path = []
        current = int(T2[np.argmax(T1[:, -1]), -1])
        path.append(end_id)
        for i in range(len(T1[0])-1, -1, -1):
            if i == 0:
                path.append(begin_id)
                continue
            print(current)
            current = int(T2[current, i])
            path.append(current)
            # current = next_path
            # print(path, i, next_path)
        path.reverse()
        print(path)
        import sys
        sys.exit()

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
                frequencies = [[0 for _ in range(len(symbols.keys())+1)] for _ in range(N)]
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