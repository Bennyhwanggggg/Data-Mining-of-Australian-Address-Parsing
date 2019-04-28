#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import math
from pprint import pprint

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
        T1 = np.array([[[0.0 for _ in range(topk)] for _ in range(len(query)+2)] for _ in range(N)])
        T2 = np.array([[[(0.0, 0.0) for _ in range(topk)] for _ in range(len(query)+2)] for _ in range(N)])

        for i in range(topk):
            T1[:, 0, i] = np.log(transition_probabilities[begin_id, :])
        prev = 0

        for i in range(1, len(query)+1):
            # i is used for index the column of dp table
            obs = query[i-1]
            for cur_state in states.keys():
                if states[cur_state] in ('BEGIN', 'END'): continue
                if i == 1:
                    for k in range(topk):
                        T1[cur_state, i, k], T2[cur_state, i, k] = (T1[cur_state, 0, k] + math.log(emission_probabilities[cur_state, obs])), begin_id
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
                    T1[cur_state, i, :] = [tmp[0] for tmp in temp[:topk]]
                    T2[cur_state, i, :] = [(tmp[1], tmp[2]) for tmp in temp[:topk]]
            prev = i

        for last_state in states.keys():
            if states[last_state] in ('BEGIN', 'END'): continue
            for k in range(topk):
                T1[last_state, -1, k], T2[last_state, -1, k] = T1[last_state, prev, k] + math.log(transition_probabilities[last_state, end_id]), last_state

        pprint(T1)
        # pprint(T2)

        last_slice = T1[:-2, -1, :]
        # print(last_slice)
        top_k_indexes = top_n_indexes(last_slice, topk)
        top_k_results = sorted([(last_slice[i, j], (i, j)) for i, j in top_k_indexes], key=lambda ele: ele[0], reverse=True)
        # print(top_k_results)
        for top_i_prob, top_i_end in top_k_results:
            path = []
            path.append(end_id)
            i, j = top_i_end
            current = T2[i, -1, j]
            for i in range(len(T2[0])-2, -1, -1):
                if i == 1:
                    path.append(begin_id)
                    break
                last_state, last_k = int(current[0]), int(current[1])
                path.append(last_state)
                current = T2[last_state, i, last_k]
            path.reverse()
            path.append(top_i_prob)
            ret.append(path)
    pprint(ret)



    #     for val, index in _get_k_largest(last_column, k):
    #         path = []
    #         score = val
    #         current = int(T2[index, -1])
    #         path.append(end_id)
    #         for i in range(len(T2[0])-2, -1, -1):
    #             if i == 0:
    #                 path.append(begin_id)
    #                 break
    #             # print(i, current)
    #             path.append(current)
    #             current = int(T2[current, i])
    #         path.reverse()
    #         path.append(score)
    #         # print(path)
    #         ret.append(path)
    # return ret


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
    top_k_res = top_k_viterbi(toy_State_File, toy_Symbol_File, toy_Query_File, 3)
    # for i in top_k_res:
    #     print(i)


if __name__ == "__main__":
    main()

