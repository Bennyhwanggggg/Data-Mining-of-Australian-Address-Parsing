# Import your files here...
import re
import numpy as np
import sys

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # Get the states and transitions from the file.
    # The states are of the format: dict[ID]: name
    # transitions are of a matrix format with ID as the indices
    # states, transitions = read_state_file(State_File)
    states, transitions = read_state_file_dict(State_File)
    # find BEGIN and END id
    begin_id, end_id = None, None
    for id in states.keys():
        if states[id] == 'BEGIN':
            begin_id = id
        if states[id] == 'END':
            end_id = id
    N = len(states.keys())
    # Get the symbols and emissions from the file.
    # The symbols are of the format: dict[name]: ID
    # emissions are of a matrix format with ID as the indices
    # symbols, emissions = read_symbol_file(Symbol_File, N)
    symbols, emissions = read_symbol_file_dict(Symbol_File, N)
    query_tokens = parse_query_file(Query_File)
    print(states, transitions)
    print(symbols, emissions)
    tokens_id = []
    # Convert each token into symbol IDs
    for query_token in query_tokens:
        tk = []
        for token in query_token:       
            symbol_id = symbols[token] if token in symbols.keys() else len(symbols.keys())  # Give UNK the last id
            tk.append(symbol_id)
        tokens_id.append(tk)

    print(tokens_id)

    # Smoothing the transition probabilities
    # transition_probabilities = np.array([[0.0 for _ in range(len(transitions[0]))] for _ in range(len(transitions))])
    # for i in range(len(transition_probabilities)):
    #     for j in range(len(transition_probabilities[0])):
    #         if states[j] == 'BEGIN':  # ignore when state to transition to is 'BEGIN' since there is no transition to it
    #             continue
    #         if states[i] == 'END':  # ignore when state to transition from is 'END' since there is no transition from it
    #             continue
    #         transition_probabilities[i, j] = (transitions[i, j] + 1) / (np.sum(transitions[i, :]) + N - 1)
    tran_prob = dict()
    for state_from in states.keys():
        if states[state_from] == 'END':
            continue
        total_time_seen = sum(transitions[state_from].values())
        if state_from not in tran_prob.keys():
            tran_prob[state_from] = dict()
        for state_to in transitions[state_from].keys():
            if state_to not in tran_prob[state_from].keys():
                tran_prob[state_from][state_to] = ((transitions[state_from][state_to] + 1)/(total_time_seen + N -1))
            else:
                tran_prob[state_from][state_to] += ((transitions[state_from][state_to] + 1)/(total_time_seen + N -1))

    # Smoothing the emission probabilities
    M = len(symbols.keys())+1 # +1 for UNK
    # emission_probabilities = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    # for i in range(N):
    #     for j in range(M):
    #         if states[i] == 'BEGIN' or states[i] == 'END':
    #             continue
    #         emission_probabilities[i, j] = (emissions[i, j] + 1) / (np.sum(emissions[i, :]) + M + 1)
    emiss_prob = dict()
    for state in states.keys():
        if states[state] == 'BEGIN' or states[state] == 'END':
            continue
        total_time_seen = sum(emissions[state].values())
        if state not in emiss_prob:
            emiss_prob[state] = dict()
        for sym in symbols.values():
            if sym not in emiss_prob[state]:
                emiss_prob[state][sym] = (emissions[state][sym] + 1)/(total_time_seen + M + 1)
            else:
                emiss_prob[state][sym] += (emissions[state][sym] + 1)/(total_time_seen + M + 1)

    print(tran_prob)
    print(emiss_prob)
    sys.exit()
    # Process each query
    for query in tokens_id:
        # setup T
        T1 = np.array([[0.0 for _ in range(M)] for _ in range(N)])
        T2 = np.array([[0.0 for _ in range(M)] for _ in range(N)])

        path = { s:[] for s in states} # init path: path[s] represents the path ends with s
        curr_pro = {}

        # Base case from starting state
        for state in states.keys():
            # initial_probabilities is transition_probabilities[begin_id, :], emission_probabilities[state, query[0]] is the probability of emission from a state to first observation
            # T1[state, begin_id] = transition_probabilities[begin_id, state]*emission_probabilities[state, query[0]]
            curr_pro[state] = transition_probabilities[begin_id, state]*emission_probabilities[state][query[0]]
        print(curr_pro)

        prev = begin_id
        for i in range(len(query)):
            observation = query[i]
            last_pro = curr_pro
            curr_pro = {}
            for state in states.keys():
                max_prob, last_state = max([(last_pro[k] * 
                                            transition_probabilities[k, state] * 
                                            emission_probabilities[state, observation], k) for k in states.keys()])
                curr_pro[state] = max_prob
                path[state].append(last_state)
            print(curr_pro)
            print(path)
            prev = observation
        # prev = begin_id
        # for i in range(len(query)):
        #     observation = query[i]
        #     for state in states.keys():
        #         T1[state, observation], T2[state, observation] = max([(T1[k, prev] * 
        #                                                                transition_probabilities[k, state] * 
        #                                                                emission_probabilities[state, observation], k) for k in states.keys()])
        #     prev = observation



        print(T1)
        print(T2)
        print('='*10)



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


def read_state_file_dict(file):
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
                frequencies = dict()
            f1, f2, f3 = map(int, line.split())
            if f1 not in frequencies.keys():
                frequencies[f1] = {f2: f3}
            else:
                if f2 not in frequencies[f1].keys():
                    frequencies[f1][f2] = f3
                else:
                    frequencies[f1][f2] += f3
    return state, frequencies


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
                frequencies = dict()
            f1, f2, f3 = map(int, line.split())
            frequencies[f1][f2] = f3
    return symbols, frequencies


def read_symbol_file_dict(file, N):
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
                frequencies = dict()
            f1, f2, f3 = map(int, line.split())
            if f1 not in frequencies.keys():
                frequencies[f1] = {f2: f3}
            else:
                if f2 not in frequencies[f1].keys():
                    frequencies[f1][f2] = f3
                else:
                    frequencies[f1][f2] += f3
    return symbols, frequencies


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