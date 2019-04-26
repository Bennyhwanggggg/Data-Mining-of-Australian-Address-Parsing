from pprint import pprint

STRIP_SYMBOL = ' \n\t'

def _print_transition(trans_table, id_to_state, state_to_id):
    # pretty print transition for debug purpose
    ret = []
    for i, row in enumerate(trans_table):
        from_state = id_to_state[i]
        ret.append([from_state])
        for j, freq in enumerate(row):
            if freq != 0:
                to_state = id_to_state[j]
                ret[-1].append((to_state, freq))
    for ln in ret:
        print(ln)
        print()
        

def parse_state_file(file):
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()
    state_num = int(lines[0].strip(STRIP_SYMBOL))
    state_to_id = dict()   # {state_name: id}
    id_to_state = dict()   # {state_name: id}
    for i in range(1, state_num+1):
        state = lines[i].strip(STRIP_SYMBOL)
        state_to_id[state] = i - 1
        id_to_state[i-1] = state
    trans_lines = lines[state_num+1:]
    transition = [[0 for _i in range(state_num)] for _j in range(state_num)]
    for line in trans_lines:
        _from, to, freq = [int(st) for st in line.strip(STRIP_SYMBOL).split(' ')]
        transition[_from][to] = freq
    return transition, state_to_id, id_to_state
        


def parse_symbol_file(file, state_num):
    lines = []
    with open(file, 'r') as f:
        lines = f.readlines()
    symbol_num = int(lines[0].strip(STRIP_SYMBOL))
    symbol_to_id = dict()   # {state_name: id}
    id_to_symbol = dict()   # {state_name: id}
    for i in range(1, symbol_num+1):
        symbol = lines[i].strip(STRIP_SYMBOL)
        symbol_to_id[symbol] = i - 1
        id_to_symbol[i-1] = symbol
    emission_lines = lines[symbol_num+1:]
    emission_list = [[0 for _i in range(symbol_num)] for _j in range(state_num)]
    emission_dict = {state_id:list() for state_id in range(state_num)}
    for line in emission_lines:
        state_id, symbol_id, freq = [int(st) for st in line.strip(STRIP_SYMBOL).split(' ')]
        emission_list[state_id][symbol_id] = freq
        emission_dict[state_id].append((symbol_id, freq))
    return emission_list, emission_dict

    

def parse_query_file(file):
    pass


def viterbi_algorithm(State_File, Symbol_File, Query_File):
    pass

def main():
    # # # Question 1
    # State_File = './dev_set/State_File'
    # Symbol_File = './dev_set/Symbol_File'
    # Query_File = './dev_set/Query_File'
    # for i in parse_query_file(Query_File):
    #     print(i)
    State_File = './toy_example/State_File'
    Symbol_File = './toy_example/Symbol_File'
    Query_File = './toy_example/Query_File'
    return viterbi_algorithm(State_File, Symbol_File, Query_File)

if __name__ == "__main__":
    main()
