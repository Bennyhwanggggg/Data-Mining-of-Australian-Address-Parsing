#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import math
from pprint import pprint

VERY_LARGE_NEG_N = -999999

def _get_k_largest(lst, k):
    """ return the k largest value and their index, reversed order
    :lst: a indexible list
    :k: num
    """
    sorted_lst = sorted([(val, index) for index, val in enumerate(lst)])
    return list(reversed(sorted_lst[-k:]))

def top_n_indexes(arr, n):
    idx = np.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    return viterbi_algorithm_helper(State_File, Symbol_File, Query_File, k=1)

# Question 1 # do not change the heading of the function
def viterbi_algorithm_helper(State_File, Symbol_File, Query_File, k):
    np.seterr(divide='ignore')
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
            # cannot go from begin to end straight away
            if states[i] == 'BEGIN' and states[j] == 'END':
                continue
            transition_probabilities[i, j] = (
                transitions[i, j] + 1) / (np.sum(transitions[i, :]) + N - 1)

    transition_probabilities = transition_probabilities[:-1, :]

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

    emission_probabilities = emission_probabilities[:-2, :]

    # Process each query
    np.set_printoptions(precision=5)
    ret = []
    for query in query_list_in_id:
        # setup dp
        T1 = np.array([[0.0 for _ in range(len(query)+2)] for _ in range(N)])
        T2 = np.array([[0.0 for _ in range(len(query)+2)] for _ in range(N)])

        T1[:, 0] = np.log(transition_probabilities[begin_id, :])
        prev = 0

        for i in range(1, len(query)+1):
            # i is used for index the column of dp table
            obs = query[i-1]
            for cur_state in states.keys():
                if states[cur_state] in ('BEGIN', 'END'): continue
                if i == 1:
                    T1[cur_state, i], T2[cur_state, i] = (T1[cur_state, 0] + math.log(emission_probabilities[cur_state, obs])), begin_id
                else:
                    T1[cur_state, i], T2[cur_state, i] = max([(T1[last_state, prev] +
                                                                math.log(transition_probabilities[last_state, cur_state]) +
                                                                math.log(emission_probabilities[cur_state, obs]),
                                                                last_state)
                                                                for last_state in states.keys() if states[last_state] not in ('BEGIN', 'END')])
            prev = i
        for last_state in states.keys():
            if states[last_state] in ('BEGIN', 'END'): continue
            T1[last_state, -1], T2[last_state, -1] = T1[last_state, prev] + math.log(transition_probabilities[last_state, end_id]), last_state

        last_column = T1[:-2, -1]
        for val, index in _get_k_largest(last_column, k):
            path = []
            score = val
            current = int(T2[index, -1])
            path.append(end_id)
            for i in range(len(T2[0])-2, -1, -1):
                if i == 0:
                    path.append(begin_id)
                    break
                # print(i, current)
                path.append(current)
                current = int(T2[current, i])
            path.reverse()
            path.append(score)
            # print(path)
            ret.append(path)
    return ret


# Question 2
# do not change the heading of the function
def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    np.seterr(divide='ignore')
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
            # cannot go from begin to end straight away
            if states[i] == 'BEGIN' and states[j] == 'END':
                continue
            transition_probabilities[i, j] = (
                transitions[i, j] + 1) / (np.sum(transitions[i, :]) + N - 1)

    transition_probabilities = transition_probabilities[:-1, :]

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

    emission_probabilities = emission_probabilities[:-2, :]

    # Process each query
    np.set_printoptions(precision=5)
    ret = []
    topk = k
    for query in query_list_in_id:
        # setup dp
        T1 = np.array([[[[] for _ in range(topk)] for _ in range(len(query)+2)] for _ in range(N)])
        T2 = np.array([[[[] for _ in range(topk)] for _ in range(len(query)+2)] for _ in range(N)])

        for i in range(topk):
            T1[:, 0, i] = [np.log(transition_probabilities[begin_id, :])]
        prev = 0

        for i in range(1, len(query)+1):
            # i is used for index the column of dp table
            obs = query[i-1]
            for cur_state in states.keys():
                if states[cur_state] in ('BEGIN', 'END'): continue
                if i == 1:
                    for k in range(topk):
                        T1[cur_state, i, k], T2[cur_state, i, k] = (T1[cur_state, 0, k] + math.log(emission_probabilities[cur_state, obs])), (begin_id, 0)

                else:
                    # Best_K_Values(t, i) = Top K over all i,preceding_state,k (emissions[i][o_t] * m[preceding_state][k] * transition[preceding_state][i])
                    temp = []
                    for last_state in states.keys():
                        if states[last_state] in ('BEGIN', 'END'):
                            continue
                        for k in range(topk):
                            p = T1[last_state, prev, k] + \
                                math.log(transition_probabilities[last_state, cur_state]) + \
                                math.log(emission_probabilities[cur_state, obs])
                            if not temp or not any([p, last_state] == [tmp[0], tmp[1]] for tmp in temp):
                                temp.append([p, last_state, k])
                    temp = sorted(temp, key=lambda ele: ele[0], reverse=True)
                    prefill = [[VERY_LARGE_NEG_N, VERY_LARGE_NEG_N, VERY_LARGE_NEG_N]]*(topk - len(temp)) if len(temp) < topk else None
                    topk_list = temp[:topk] if prefill is None else temp + prefill
                    T1[cur_state, i, :] = [tmp[0] for tmp in topk_list]
                    T2[cur_state, i, :] = [(tmp[1], tmp[2]) for tmp in topk_list]
            prev = i

        for last_state in states.keys():
            if states[last_state] in ('BEGIN', 'END'): continue
            for k in range(topk):
                T1[last_state, -1, k], T2[last_state, -1, k] = T1[last_state, prev, k] + math.log(transition_probabilities[last_state, end_id]), (last_state, k)

        # pprint(T1)
        # print('T2 is')
        # pprint(T2)

        last_slice = T1[:-2, -1, :]
        top_k_indexes = top_n_indexes(last_slice, last_slice.shape[0]*last_slice.shape[1])
        top_k_results = sorted([(last_slice[i, j], (i, j)) for i, j in top_k_indexes], key=lambda ele: ele[0], reverse=True)
        current_ret = []
        pprint(last_slice)
        for top_i_prob, top_i_end in top_k_results:
            path = []
            path.append(end_id)
            i, j = top_i_end
            current = T2[i, -1, j]
            fail_path = False
            for i in range(len(T2[0])-2, -1, -1):
                if i == 0:
                    path.append(begin_id)
                    break
                last_state, last_k = int(current[0]), int(current[1])
                if last_k == VERY_LARGE_NEG_N:
                    fail_path = True
                    break
                path.append(last_state)
                current = T2[last_state, i, last_k]
            if fail_path:
                continue
            path.reverse()
            path.append(top_i_prob)
            current_ret.append(path)
        current_ret = sorted(current_ret, key=lambda x: (-x[-1], [x[a] for a in range(len(x)-1, -1, -1)]))
        ret.extend(current_ret[:topk])
    return ret


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
    delimiters = ',()/-&'
    tokens = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            token = []
            queries = line.split()
            for q in queries:
                tk = re.split(r'(\,|\(|\)|\/|-|\&)', q)
                for t in tk:
                    if t != '':
                        token.append(t)
            tokens.append(token)
    return tokens


def main():
    # Question 1
    State_File = './dev_set/State_File'
    Symbol_File = './dev_set/Symbol_File'
    Query_File = './dev_set/Query_File'
    # for i in parse_query_file(Query_File):
    #     print(i)
    toy_State_File = './toy_example/State_File'
    toy_Symbol_File = './toy_example/Symbol_File'
    toy_Query_File = './toy_example/Query_File'
    # viterbi_result = viterbi_algorithm(State_File, Symbol_File, Query_File)
    # top_k_res = top_k_viterbi(State_File, Symbol_File, Query_File, 3)
    top_k_res = top_k_viterbi(toy_State_File, toy_Symbol_File, toy_Query_File, 20)
    top_20_res = [[3, 0, 0, 1, 2, 4, -9.843403381747937], [3, 0, 0, 0, 2, 4, -10.131085454199718], [3, 0, 0, 0, 0, 4, -10.20007832568667], [3, 1, 2, 1, 2, 4, -10.382399882480625], [3, 0, 2, 1, 2, 4, -10.536550562307884], [3, 0, 0, 0, 1, 4, -10.641911077965709], [3, 0, 0, 1, 1, 4, -10.913844793449352], [3, 2, 1, 1, 2, 4, -10.942015670416048], [3, 0, 0, 2, 1, 4, -11.047376186073874], [3, 2, 0, 1, 2, 4, -11.096166350243307], [3, 0, 1, 1, 2, 4, -11.096166350243307], [3, 2, 0, 0, 2, 4, -11.383848422695086], [3, 2, 0, 0, 0, 4, -11.452841294182038], [3, 1, 2, 1, 1, 4, -11.452841294182038], [3, 1, 1, 1, 2, 4, -11.50163145835147], [3, 2, 1, 2, 1, 4, -11.58637268680656], [3, 0, 2, 1, 1, 4, -11.606991974009297], [3, 2, 2, 1, 2, 4, -11.635162850975993], [3, 0, 1, 2, 1, 4, -11.74052336663382], [3, 1, 0, 1, 2, 4, -11.78931353080325], [3, 2, 1, 2, 4, -9.397116279119517], [3, 0, 0, 2, 4, -9.551266958946776], [3, 0, 1, 2, 4, -9.551266958946776], [3, 0, 0, 0, 4, -9.620259830433728], [3, 1, 2, 1, 4, -9.907941902885508], [3, 1, 1, 2, 4, -9.956732067054942], [3, 0, 0, 1, 4, -10.062092582712767], [3, 0, 2, 1, 4, -10.062092582712767], [3, 2, 1, 1, 4, -10.467557690820932], [3, 0, 1, 1, 4, -10.62170837064819], [3, 1, 2, 2, 4, -10.649879247614885], [3, 2, 0, 2, 4, -10.804029927442144], [3, 0, 2, 2, 4, -10.804029927442144], [3, 2, 0, 0, 4, -10.873022798929096], [3, 1, 2, 0, 4, -10.873022798929096], [3, 0, 2, 0, 4, -11.027173478756353], [3, 1, 1, 1, 4, -11.027173478756355], [3, 2, 2, 1, 4, -11.160704871380876], [3, 2, 0, 1, 4, -11.314855551208135], [3, 1, 0, 2, 4, -11.497177108002088]]
    i = 0
    pprint(top_k_res)
    for res, expected in zip(top_k_res, top_20_res):
        if res != expected:
            print('{}: expected: {}, got: {}'.format(i, expected, res))
        i+=1


if __name__ == "__main__":
    main()

