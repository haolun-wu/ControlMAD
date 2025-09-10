from project_types import base_config
from utility import Utility

class Validator:
    def __init__(self, config: base_config):
        self.config = config
        self.utility = Utility()

    def validate_single_game(self, schemas: list, statements: list):
        size = self.config.game_size
        num_spy = self.config.num_spy
        num_hint = self.config.num_hint
        
        player_list = [i for i in range(size)]
        solutions = [] # Record all possible solutions for the game.
        # Enumerate all possible spy lists to start with.
        all_possible_spy_lists = self.utility.enumerate_sublists(player_list, num_spy)
        for spy_list in all_possible_spy_lists:
            # For each possible spy list, check if it can be extended to a valid solution.
            # First find the non-spy list.
            KK_list = []
            for i in player_list:
                if i not in spy_list:
                    KK_list.append(i)
            for num_knight in range(0, size - num_spy + 1):
                if num_knight == 0:
                    knight_list = []
                    knave_list = KK_list
                    if self.validate_candidate_solution(spy_list, knight_list, knave_list, schemas, statements):
                        solution = []
                        for i in range(size):
                            if i in knight_list:
                                solution.append(1)
                            elif i in knave_list:
                                solution.append(2)
                            else:
                                solution.append(3)
                        solutions.append(solution)
                else:
                    # Enumerate all possible knight lists.
                    all_possible_knight_lists = self.utility.enumerate_sublists(KK_list, num_knight)
                    for knight_list in all_possible_knight_lists:
                        knave_list = []
                        for i in KK_list:
                            if i not in knight_list:
                                knave_list.append(i)
                        if self.validate_candidate_solution(spy_list, knight_list, knave_list, schemas, statements):
                            solution = []
                            for i in range(size):
                                if i in knight_list:
                                    solution.append(1)
                                elif i in knave_list:
                                    solution.append(2)
                                else:
                                    solution.append(3)
                            solutions.append(solution)
            pass
        return solutions
    
    def validate_candidate_solution(self, spy_list: list, knight_list: list, knave_list: list, schemas: list, statements: list):
        size = self.config.game_size
        # Validate all knight statements. They should be true.
        tf_list = []
        length = len(schemas)
        # Check the truthfulness of all statements in order. This is because some statements depend on the truthfulness of previous statements.
        for i in range(length):
            tf = (self.validate_single_statement(spy_list, knight_list, knave_list, schemas[i], statements[i], tf_list))
            tf_list.append(tf)
        # Check the truthfulness of all knight statements. They should be true.
        for i in knight_list:
            if not tf_list[i]:
                return False
        # Check the truthfulness of all knave statements. They should be false.
        for i in knave_list:
            if tf_list[i]:
                return False
        # Check the truthfulness of all hints from GameManager. They should be true.
        for i in range(size, length):
            if not tf_list[i]:
                return False
        return True

    def validate_single_statement(self, spy_list: list, knight_list: list, knave_list: list, type: int, statement: list, tf_list: list):
        role_map = {
            1: knight_list,
            2: knave_list,
            3: spy_list
        }
        # 1-3: Among (set) there are (number) people who are (role). statement: [set, n, role]
        if type in [1, 2, 3]:
            if statement[1] == self.utility.count_intersection(role_map[statement[2]], statement[0]):
                return True
            return False
        # 4-5: Among (set) there are (number) people who says (TF). statement: [set, n, tf]
        if type in [4, 5]:
            count = 0
            for i in statement[0]:
                # statement[2] = 1 means telling the truth, 2 means lying.
                if tf_list[i] == (statement[2] == 1):
                    count += 1
            if count == statement[1]:
                return True
            return False
        # 6-7: People from (set) have same role. statement: [set]
        if type in [6, 7]:
            if set(statement[0]).issubset(set(knight_list)):
                return True
            if set(statement[0]).issubset(set(knave_list)):
                return True
            if set(statement[0]).issubset(set(spy_list)):
                return True
            return False
        # 8-9: People from (set) have have same TF. statement: [set]
        if type in [8, 9]:
            tf_base = tf_list[statement[0][0]]
            for i in statement[0]:
                if tf_list[i] != tf_base:
                    return False
            return True
        # 10: People from (set) have different roles. statement: [set]
        if type == 10:
            if set(statement[0]).issubset(set(knight_list)):
                return False
            if set(statement[0]).issubset(set(knave_list)):
                return False
            if set(statement[0]).issubset(set(spy_list)):
                return False
            return True
        # 11-12: Among (set) the number of (role) is (EO). statement: [set, role, eo]
        if type in [11, 12]:
            return ((self.utility.count_intersection(role_map[statement[1]], statement[0])) % 2) == (statement[2] % 2)
        # 13-14: Among (set) the number of people saying (TF) is (EO). statement: [set, tf, eo]
        if type in [13, 14]:
            count = 0
            for i in statement[0]:
                if tf_list[i] == (statement[1] == 1):
                    count += 1
            return (count % 2) == (statement[2] % 2)
        # 15: Among the following two statements, only one is true. statement: [[new_type_1, statement_1], [new_type_2, statement_2]]
        if type == 15:
            # print ("*"*50)
            # print ("Checking statement of type 15")
            result_1 = self.validate_single_statement(spy_list, knight_list, knave_list, statement[0][0], statement[0][1], tf_list)
            result_2 = self.validate_single_statement(spy_list, knight_list, knave_list, statement[1][0], statement[1][1], tf_list)
            # print ("First statement: ", result_1)
            #print ("Second statement: ", result_2)
            result = (result_1 != result_2)
            #print ("Result: ", result)
            # print ("*"*50)
            return result

if __name__ == "__main__":
    pass