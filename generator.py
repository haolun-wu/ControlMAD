import random
import json
from project_types import player, base_config

class Generator:
    def __init__(self, config: base_config):
        self.name_pool = config.name_pool
        self.role_map = config.role_map
        self.tf_map = config.tf_map
        self.eo_map = config.eo_map
        self.number_map = config.number_map
        self.size = config.game_size
        self.num_spy = config.num_spy
        self.num_hint = config.num_hint
    
    def generate_single_game_abstract(self):
        schemas = []
        statements = []
        size = self.size
        num_spy = self.num_spy
        num_hint = self.num_hint
        """
        We have the following shcemas:
        1-3: Among (set) there are (number) people who are (role). statement: [set, n, role]
        4-5: Among (set) there are (number) people who says (TF). statement: [set, n, tf]
        6-7: People from (set) have same role. statement: [[set]]
        8-9: People from (set) have have same TF. statement: [[set]]
        10: People from (set) do not all have same role. statement: [[set]]
        11-12: Among (set) the number of (role) is (EO). statement: [set, role, eo]
        13-14: Among (set) the number of people saying (TF) is (EO). statement: [set, tf, eo]
        15: Among the following two statements, only one is true:. statement: [[new_type_1, statement_1], [new_type_2, statement_2]]
        """
        for id in range(size):
            type = self.generate_random_type(id)
            
            schemas.append(type)
            statements.append(self.generate_single_statement_abstract(type, id))
            
        # Setup a GM who gives hints. Should be always true.
        # First, alwasy reveals the number of spies.
        schemas.append(1)
        statement_about_spy = [[i for i in range(size)], num_spy, 3]
        statements.append(statement_about_spy)
        
        for i in range(num_hint):
            type = random.randint(1,10)
            schemas.append(type)
            statement = self.generate_single_statement_abstract(type, size)
            # Make sure extra hints are not about the number of spies, as it is already revealed.
            while (type in [1, 2, 3]):
                if (len(statement[0]) != size):
                    break
                if statement[2] != 3:
                    break
                statement = self.generate_single_statement_abstract(type, size)
            statements.append(statement)

        return schemas, statements
    
    def generate_random_type(self, id: int):
        type = random.randint(1, 15)
        # The first people cannot talk about other people's truthfulness.
        # The first three people cannot talk about the parity of the number of people who say a certain truthfulness, since they should only talk about people prior to them, and should have a set of at least 3 people.
        while ((id == 0 and type in [4, 5]) or (id < 3 and type in [13, 14])) or (id < 2 and type in [8, 9]):
                type = random.randint(1, 15)
        return type

    
    def generate_single_statement_abstract(self, type: int, id: int):
        size = self.size
        num_spy = self.num_spy
        # When talking about other people's truthfulness, we only talk about the people before the current person. This is to avoid circular reasoning.
        if (type in [4, 5]) or (type in [8, 9]) or (type in [13, 14]):
            pool = [i for i in range(id)]
        else:
            pool = [i for i in range(size)]
        # 1-3: Among (set) there are (number) people who are (role). statement: [set, n, role]
        if type in [1, 2, 3]:
                set_size = random.randint(1, size)
                set = random.sample(pool, set_size)
                role = random.randint(1, 3)
                # If the role is spy, we need to consider the number of spies in the game.
                if role == 3:
                    n = random.randint(1, min(num_spy, set_size))
                else:
                    n = random.randint(1, set_size)
                return [set, n, role]
        # 4-5: Among (set) there are (number) people who says (TF). statement: [set, n, tf]
        if type in [4, 5]:
            # When talking about other people's truthfulness, we only talk about the people before the current person. This is to avoid circular reasoning.
            set_size = random.randint(1, id)
            set = random.sample(pool, set_size)
            n = random.randint(1, set_size)
            tf = random.randint(1, 2)
            return [set, n, tf]
        # 6-7: People from (set) have same role. statement: [set]
        if type in [6, 7]:
            # When claiming that a set of people have the same role, the set must have at least 2 people.
            set_size = random.randint(2, size - 1)
            set = random.sample(pool, set_size)
            return [set]
        # 8-9: People from (set) have have same TF. statement: [set]
        # When claiming that a set of people have the same truthfulness, the set must have at least 2 people. Also, can only talk about people prior to the current one.
        if type in [8, 9]:
            set_size = random.randint(2, id)
            set = random.sample(pool, set_size)
            return [set]
        # 10: People from (set) do not all have same role. statement: [set]
        if type in [10]:
            set_size = random.randint(2, size)
            set = random.sample(pool, set_size)
            return [set]
        # 11-12: Among (set) the number of (role) is (EO). statement: [set, role, eo]
        if type in [11, 12]:
            # When talking about the parity of the number of people who have a certain role, the set must have at least 3 people.
            set_size = random.randint(3, size)
            set = random.sample(pool, set_size)
            role = random.randint(1, 3)
            eo = random.randint(1, 2)
            return [set, role, eo]
        # 13-14: Among (set) the number of people saying (TF) is (EO). statement: [set, tf, eo]
        if type in [13, 14]:
            # When talking about the parity of the number of people who say a certain truthfulness, the set must have at least 3 people.
            set_size = random.randint(3, id)
            set = random.sample(pool, set_size)
            tf = random.randint(1, 2)
            eo = random.randint(1, 2)
            return [set, tf, eo]
        # 15: Among the following two statements, only one is true:. statement: [[new_type_1, statement_1], [new_type_2, statement_2]]
        if type == 15:
            # type 15 requires two statements.
            new_type_1 = self.generate_random_type(id)
            # The new type cannot be 15.
            while new_type_1 == 15:
                new_type_1 = self.generate_random_type(id)
            # The new type cannot be 15.
            new_type_2 = self.generate_random_type(id)
            while new_type_2 == 15:
                new_type_2 = self.generate_random_type(id)
            return [[new_type_1, self.generate_single_statement_abstract(new_type_1, id)], [new_type_2, self.generate_single_statement_abstract(new_type_2, id)]]

    def generate_single_statement_text(self, id: int, player_list: list, type: int, statement_abstract: list):
        size = self.size
        statement_text = ""
        # 1-3: Among (set) there are (number) people who are (role). statement: [set, n, role]
        if type in [1, 2, 3]:
            # If there is only one person involved in the statement.
            if len(statement_abstract[0]) == 1:
                # If this person is the person making the claim, the claim is about himself.
                if id == statement_abstract[0][0]:
                    return "I am a " + self.role_map[statement_abstract[2]] + "."
                return self.compose_people_list(player_list, statement_abstract[0], id) + " is a " + self.role_map[statement_abstract[2]] + "."
            # If this is a claim about all people
            # If all people in the set have the same role.
            if statement_abstract[1] == len(statement_abstract[0]):
                # If there are two people, use "both". 
                if statement_abstract[1] == 2 :
                    if statement_abstract[2] == 3:
                        return self.compose_people_list(player_list, statement_abstract[0], id) + " are both spies."
                    return self.compose_people_list(player_list, statement_abstract[0], id) + " are both " + self.role_map[statement_abstract[2]] + "s."
                if statement_abstract[1] == size:
                    if statement_abstract[2] == 3:
                        return "All are spies."
                    return "All are " + self.role_map[statement_abstract[2]] + "s."
                if statement_abstract[2] == 3:
                    return self.compose_people_list(player_list, statement_abstract[0], id) + " are all spies."
                return self.compose_people_list(player_list, statement_abstract[0], id) + " are all " + self.role_map[statement_abstract[2]] + "s."
            if len(statement_abstract[0]) == size:
                statement_text += "Among all players, there "
            else:
                statement_text += "Among " + self.compose_people_list(player_list, statement_abstract[0], id) + ", there "
            # If there is only one person who has the role.
            if statement_abstract[1] == 1:
                return statement_text + "is exactly one " + self.role_map[statement_abstract[2]] + "."
            # If there are multiple people who have the role.
            if statement_abstract[2] == 3:
                return statement_text + "are exactly " + self.number_map[statement_abstract[1]] + " spies."
            return statement_text + "are exactly " + self.number_map[statement_abstract[1]] + " " + self.role_map[statement_abstract[2]] + "s."
        # 4-5: Among (set) there are (number) people who says (TF). statement: [set, n, tf]
        if type in [4, 5]:
            # If there is only one person involved in the statement.
            if len(statement_abstract[0]) == 1:
                # If this person is the person making the claim, the claim is about himself.
                if id == statement_abstract[0][0]:
                    return "I am " + self.tf_map[statement_abstract[2]] + "."
                return self.compose_people_list(player_list, statement_abstract[0], id) + " is " + self.tf_map[statement_abstract[2]] + "."
            # If this is a claim about all people
            # If all people in the set have the same truthfulness.
            if statement_abstract[1] == len(statement_abstract[0]):
                # If there are two people, use "both". 
                if statement_abstract[1] == 2:
                    return self.compose_people_list(player_list, statement_abstract[0], id) + " are both " + self.tf_map[statement_abstract[2]] + "."
                if statement_abstract[1] == size:
                    return "All are " + self.tf_map[statement_abstract[2]] + "."
                return self.compose_people_list(player_list, statement_abstract[0], id) + " are all " + self.tf_map[statement_abstract[2]] + "."
            if len(statement_abstract[0]) == size:
                statement_text += "Among all players, "
            else:
                statement_text += "Among " + self.compose_people_list(player_list, statement_abstract[0], id) + ", "
            # If there is only one person who has the role.
            if statement_abstract[1] == 1:
                return statement_text + "exactly one person is " + self.tf_map[statement_abstract[2]] + "."
            # If there are multiple people who have the role.
            return statement_text + "exactly " + self.number_map[statement_abstract[1]] + " people are " + self.tf_map[statement_abstract[2]] + "."
        # 6-7: People from (set) have same role. statement: [set]
        if type in [6, 7]:
            if len(statement_abstract[0]) == 2: 
                return self.compose_people_list(player_list, statement_abstract[0], id) + " have the same role."
            return self.compose_people_list(player_list, statement_abstract[0], id) + " all have the same role."
        # 8-9: People from (set) have have same TF. statement: [set]
        if type in [8, 9]:
            if len(statement_abstract[0]) == 2:
                return self.compose_people_list(player_list, statement_abstract[0], id) + " are either both telling truth or both lying."
            return self.compose_people_list(player_list, statement_abstract[0], id) + " are either all telling truth or all lying."
        # 10: People from (set) do not all have same role. statement: [set]
        if type == 10:
            if len(statement_abstract[0]) == 2:
                return self.compose_people_list(player_list, statement_abstract[0], id) + " have different roles."
            return self.compose_people_list(player_list, statement_abstract[0], id) + " do not all have the same role."
        # 11-12: Among (set) the number of (role) is (EO). statement: [set, role, eo]
        if type in [11, 12]:
            if len(statement_abstract[0]) == size:
                statement_text += "Among all players, "
            else:
                statement_text += "Among " + self.compose_people_list(player_list, statement_abstract[0], id) + ", "
            if statement_abstract[1] == 3:
                return statement_text + "the number of spies is " + self.eo_map[statement_abstract[2]] + "."
            return statement_text + "the number of " + self.role_map[statement_abstract[1]] + "s is " + self.eo_map[statement_abstract[2]] + "."
        # 13-14: Among (set) the number of people saying (TF) is (EO). statement: [set, tf, eo]
        if type in [13, 14]:
            if len(statement_abstract[0]) == size:
                statement_text += "Among all players, "
            else:
                statement_text += "Among " + self.compose_people_list(player_list, statement_abstract[0], id) + ", "
            return statement_text + "the number of people who are " + self.tf_map[statement_abstract[1]] + " is " + self.eo_map[statement_abstract[2]] + "."
        # 15: Among the following two statements, only one is true:. statement: [[new_type_1, statement_1], [new_type_2, statement_2]]
        if type == 15:
            statement_text_1 = self.generate_single_statement_text(id, player_list, statement_abstract[0][0], statement_abstract[0][1])
            statement_text_2 = self.generate_single_statement_text(id, player_list, statement_abstract[1][0], statement_abstract[1][1])
            return f"Among the following two statements, exactly one is true: \n (1). {statement_text_1}\n (2). {statement_text_2}"
        
    def compose_people_list(self, player_list: list, abstract_list: list, id: int):
        if id in abstract_list:
            new_abstract_list = [i for i in abstract_list if i != id]
            new_abstract_list.append(id)
        else:
            new_abstract_list = abstract_list

        if len(new_abstract_list) == 1:
            return player_list[new_abstract_list[0]]
        if len(new_abstract_list) == 2:
            if new_abstract_list[1] == id:
                return player_list[new_abstract_list[0]] + " and I"
            return player_list[new_abstract_list[0]] + " and " + player_list[new_abstract_list[1]]
        content = ""
        for (i, name_id) in enumerate(new_abstract_list):
            if i < len(new_abstract_list) - 1:
                content += player_list[name_id] + ", "
            else:
                if name_id == id:
                    content += "and I"
                else:
                    content += "and " + player_list[name_id]
        return content
    
    def generate_single_game_text(self, schemas: list, statements: list):
        num_hint = self.num_hint
        # Error handling.
        if (len(schemas) <= 1) or (len(statements) <= 1) or (len(schemas) != len(statements)):
            return None
        size = self.size
        player_name_list = random.sample(self.name_pool, size)
        players = []
        for i in range(size):
            players.append(
                player(
                    name = player_name_list[i],
                    statement = self.generate_single_statement_text(i, player_name_list, schemas[i], statements[i]),
                    description = ""
                    )
                )
        if num_hint == 0:
            gm_statement = "I am the game manager and here is a hint for you: " + self.generate_single_statement_text(size, player_name_list, schemas[size], statements[size])
        else:
            gm_statement = "I am the game manager and here are some hints for you:"
            gm_statement += "\n 1. " + self.generate_single_statement_text(size, player_name_list, schemas[size], statements[size])
        for i in range(num_hint):
            gm_statement += "\n " + str(i + 2) + ". " + self.generate_single_statement_text(size + i + 1, player_name_list, schemas[size + i + 1], statements[size + i + 1])
        players.append(
            player(
                name = "GameManager",
                statement = gm_statement,
                description = ""
                )
            )
        return players
    def generate_single_game_warp(self):
        schemas, statements = self.generate_single_game_abstract()
        players = self.generate_single_game_text(schemas, statements)
        return schemas, statements, players

    

    

if __name__ == "__main__":
    pass
