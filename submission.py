# Import your files here...
import re
import numpy as np
import math
from pprint import pprint

# Question 1


# do not change the heading of the function
def viterbi_algorithm(State_File, Symbol_File, Query_File):
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
    query_list = parse_query_file(Query_File)
    query_list_in_id = []
    # Convert each token into symbol IDs
    for query_in_token in query_list:
        tk = []
        for token in query_in_token:
            symbol_id = symbols[token] if token in symbols.keys() else len(
                symbols.keys())  # Give UNK the last id
            tk.append(symbol_id)
        query_list_in_id.append(tk)

    # Smoothing the transition probabilities
    transition_probabilities = np.array(
        [[0.0 for _ in range(len(transitions[0]))] for _ in range(len(transitions))])
    for i in range(len(transition_probabilities)):
        for j in range(len(transition_probabilities[0])):
            # ignore when state to transition to is 'BEGIN' since there is no transition to it
            if states[j] == 'BEGIN':
                continue
            # ignore when state to transition from is 'END' since there is no transition from it
            if states[i] == 'END':
                continue
            # cannot go from begin to end straight away?
            if states[i] == 'BEGIN' and states[j] == 'END':
                continue
            transition_probabilities[i, j] = (
                transitions[i, j] + 1) / (np.sum(transitions[i, :]) + N - 1)

    # Smoothing the emission probabilities
    M = len(symbols.keys())+1  # +1 for UNK
    emission_probabilities = np.array(
        [[0.0 for _ in range(M)] for _ in range(N)])
    for i in range(N):
        for j in range(M):
            if states[i] == 'BEGIN' or states[i] == 'END':
                continue
            emission_probabilities[i, j] = (
                emissions[i, j] + 1) / (np.sum(emissions[i, :]) + M)
    # return query_list_in_id, transition_probabilities, emission_probabilities, states # debug with ipython
    # return locals()

    # Process each query
    # np.set_printoptions(precision=3)
    ret = []
    for query in query_list_in_id:
        # setup dp
        T1 = np.array([[0.0 for _ in range(len(query)+2)] for _ in range(N)])
        T2 = np.array([[0.0 for _ in range(len(query)+2)] for _ in range(N)])

        # starting transition probabilities
        T1[:, 0] = np.log(transition_probabilities[begin_id, :])
        T2[:, 0] = begin_id
        prev = 0

        for i in range(1, len(query)+1):
            # i is used for index the column of dp table
            obs = query[i-1]
            for cur_state in states.keys():
                if states[cur_state] in ('BEGIN', 'END'): continue
                T1[cur_state, i], T2[cur_state, i] = max([(T1[last_state, prev] +
                                                           math.log(transition_probabilities[last_state, cur_state]) +
                                                           math.log(emission_probabilities[cur_state, obs]),
                                                           last_state)
                                                          for last_state in states.keys() if states[last_state] not in ('BEGIN', 'END')])
            prev = i
        # transite to END?
        for last_state in states.keys():
            if states[last_state] in ('BEGIN', 'END'): continue
            T1[last_state, -1], T2[last_state, -1] = T1[last_state, prev] + math.log(transition_probabilities[last_state, end_id]), last_state
                                          
        print("dp table:")
        pprint(T1)
        print("come from table:")
        pprint(T2)

        # backtract to get path?
        # print("debug", np.argmax(T1[:-2, -1]))
        path = []
        score = max(T1[:-2, -1])
        current = int(T2[np.argmax(T1[:-2, -1]), -1])
        path.append(end_id)
        for i in range(len(T2[0])-2, -1, -1):
            if i == 0:
                path.append(begin_id)
                break
            pprint(current)
            path.append(current)
            current = int(T2[current, i])
            # current = next_path
            # print(path, i, next_path)
        path.reverse()
        print("path is")
        pprint(path)
        path.append(score)
        ret.append(path)
        return ret


# Question 2


# do not change the heading of the function
def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    pass  # Replace this line with your implementation...


# Question 3 + Bonus
# do not change the heading of the function
def advanced_decoding(State_File, Symbol_File, Query_File):
    pass  # Replace this line with your implementation...


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
                frequencies = [
                    [0 for _ in range(len(state.keys()))] for _ in range(len(state.keys()))]
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
                frequencies = [
                    [0 for _ in range(len(symbols.keys())+1)] for _ in range(N)]
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


def main():
    # Question 1
    State_File = './toy_example/State_File'
    Symbol_File = './toy_example/Symbol_File'
    Query_File = './toy_example/Query_File'
    viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
    return viterbi_result


if __name__ == "__main__":
    main()
