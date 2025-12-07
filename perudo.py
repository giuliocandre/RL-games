import random
import math
MAX_PLAYERS = 4
DICES_PER_PLAYER = 5
DOUBT_ACTION = (-1, -1)
JOLLY_FACE = 1

'''
RL Agent that learns how to play Perudo.
State space: 
        - total number of dices in game
        - player's own dices (as a list of counts for each face)
        - last bid made (quantity and face)
Action space:
        - make a bid (quantity and face) with constraints
        - doubt the last bid

Reward structure:
        - +1 for winning a game
        - 0 otherwise
'''

# Generate all possible dice face counts for a given number of dices
all_possible_dice_counts_table = {0 : [[0] * 6]}

for tot in range(1, DICES_PER_PLAYER + 1):
    for prev in all_possible_dice_counts_table[tot - 1]:
        for n in range(6):
            fcs = prev.copy()
            fcs[n] += 1
            all_possible_dice_counts_table[tot] = all_possible_dice_counts_table.get(tot, []) + [fcs]


        

def generate_state_space(n_players=MAX_PLAYERS):
    state_space = {}
    for tot_dices in range(2, n_players * DICES_PER_PLAYER + 1):
        cur_state = [0] * (1 + 2 + 6) # tot_dices + last bid (quantity, face) + own dices (6 faces) 
        cur_state[0] = tot_dices
        for n_dices in range(1, 6):
            for dice_confs in all_possible_dice_counts_table.get(n_dices, []):
                cur_state[3:] = dice_confs
                for last_bid_quantity in range(0, tot_dices + 1):
                    for last_bid_face in range(0, 7):
                        cur_state[1] = last_bid_quantity
                        cur_state[2] = last_bid_face
                        state_space[tuple(cur_state)] = 0
    return state_space
        
def possible_actions_from_bid(tot_dices, last_bid_quantity, last_bid_face):
        
        actions = set()
        
        min_next_bid_q = max(1, last_bid_quantity)
        min_next_bid_f = max(1, last_bid_face)
        # doubt action only if there was a last bid
        if last_bid_quantity > 0:
            actions.add(DOUBT_ACTION)

        # last bid was jolly bid?
        if last_bid_face == JOLLY_FACE:
            min_next_bid_q = 2*last_bid_quantity + 1
            min_next_bid_f = 2

        # special jolly bid            
        for bid_quantity in range((min_next_bid_q // 2) + (min_next_bid_q % 2), tot_dices + 1):
            actions.add((bid_quantity, JOLLY_FACE))

        # possible bids
        for bid_quantity in range(min_next_bid_q, tot_dices + 1):
            for bid_face in range(min_next_bid_f, 7):
                if bid_quantity == min_next_bid_q and bid_face == last_bid_face:
                    continue
                actions.add((bid_quantity, bid_face))
        return list(actions)

def generate_action_space(state_space):
    action_space = {}
    for state in state_space.keys():
        tot_dices = state[0]
        last_bid_quantity = state[1]
        last_bid_face = state[2]
        # Skip if already generated
        if (action_space.get((tot_dices, last_bid_quantity, last_bid_face))):
            continue
        actions = possible_actions_from_bid(tot_dices, last_bid_quantity, last_bid_face)
        action_space[(tot_dices, last_bid_quantity, last_bid_face)] = actions
    return action_space

def generate_policy_map(state_space, action_space):
    policy_map = {}
    for state in state_space.keys():
        actions = action_space[(state[0], state[1], state[2])]
        for a in actions:
            policy_map[state, a] = 1 # Equal weight per action
    return policy_map

''' Generate all possible states, actions and initialize policy map 
    NOT NEEDED
'''
#STATE_SPACE = generate_state_space()
#ACTION_SPACE = generate_action_space(STATE_SPACE)
#POLICY_MAP = generate_policy_map(STATE_SPACE, ACTION_SPACE)
POLICY_MAP = {}

def policy(state, epsilon=0.05):
    actions = possible_actions_from_bid(state[0], state[1], state[2])
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        # Choose action with highest weight
        max_ = -math.inf
        best_action = None
        for a in actions:
            action_value = POLICY_MAP.get((state, a), 1)
            if action_value > max_:
                max_ = action_value
                best_action = a

        return best_action
        
        


def render_dices(dices):
    for idx, count in enumerate(dices):
        if count > 0:
            def f2c(f):
                return "Jolly" if f == JOLLY_FACE else str(f)
            print(f"    {count} of {f2c(idx + 1)}")

def is_legal_bid(last_bid, new_bid):
    last_bid_quantity, last_bid_face = last_bid
    new_bid_quantity, new_bid_face = new_bid

    if last_bid_quantity == 0 and last_bid_face == 0:
        return True

    if new_bid == DOUBT_ACTION:
        return last_bid_quantity > 0

    min_bid_q = last_bid_quantity
    min_bid_f = last_bid_face
    if last_bid_face == JOLLY_FACE:
        min_bid_q = 2*last_bid_quantity
        min_bid_f = 1

    if new_bid_quantity > min_bid_q and new_bid_face >= min_bid_f:
        return True
    
    if new_bid_quantity >= min_bid_q and new_bid_face > min_bid_f:
        return True


    if new_bid_face == JOLLY_FACE and new_bid_quantity >= ((min_bid_q % 2) + (min_bid_q // 2)):
        return True

    return False

class Player():
    def __init__(self, name, n_dices=DICES_PER_PLAYER):
        self.name = name
        self.n_dices = n_dices
        self.history = []
        self.new_round()
    
    def new_round(self):
        self.dices = random.choice(all_possible_dice_counts_table[self.n_dices])

    def lose_dice(self):
        self.n_dices = max(0, self.n_dices - 1)
        self.flush_history(-1)
        if self.n_dices == 0:
            self.lose()
    
    def flush_history(self, final_reward):
        reward = final_reward
        for state, action in self.history[::-1]:
            cur = POLICY_MAP.get((state, action), 1)
            POLICY_MAP[state, action] = cur + 0.1 * (reward - cur)
            reward = POLICY_MAP[state, action]

    def lose(self):
        self.flush_history(-10)

    def win(self):
        self.flush_history(10)

    def is_alive(self):
        return self.n_dices > 0
    
    def track_action(self, total_dices, last_bid, action):
        state = [0] * (1 + 2 + 6)
        state[0] = total_dices
        state[1] = last_bid[0]
        state[2] = last_bid[1]
        state[3:] = self.dices
        state = tuple(state)
        self.history.append((state, action))
    
    def make_action(self, *args):
        pass

class RLPlayer(Player):
    GAMES = 0
    WINS = 0

    @staticmethod
    def get_stats():
        return RLPlayer.WINS / RLPlayer.GAMES if RLPlayer.GAMES > 0 else 0
    
    def track_stats(self):
        self.track_stats_ = True
        RLPlayer.GAMES += 1

    def make_action(self, total_dices, last_bid):
        state = [0] * (1 + 2 + 6)
        state[0] = total_dices
        state[1] = last_bid[0]
        state[2] = last_bid[1]
        state[3:] = self.dices
        state = tuple(state)
        action = policy(state)
        return action
    
    def win(self):
        super().win()
        if self.track_stats_:
            RLPlayer.WINS += 1

    def __init__(self, name, n_dices=DICES_PER_PLAYER):
        super().__init__(name, n_dices)
        self.track_stats_ = False
        
'''
Player that uses a static strategy:
- other_dices = total_dices - self.n_dices
- estimate for every face = own_dices[face - 1] + own_dices[0] + other_dices / 3 
- estimate for jolly = own_dices[0] + other_dices / 6
- find if there is a possible action that is lower than estimate, pick the highest one
- else, doubt
'''
class RoboPlayer(Player):
    def new_round(self):
        super().new_round()
        #print(f"{self.name} dices: ")
        #render_dices(self.dices)

    def get_estimate_vector(self, total_dices):
        estimate = [0] * 6
        other_dices = total_dices - self.n_dices
        for face in range(1, 7):
            if face == JOLLY_FACE:
                estimate[face - 1] = self.dices[0] + other_dices / 6
            else:
                estimate[face - 1] = self.dices[face - 1] + self.dices[0] + other_dices / 3
        return estimate
    
    def aggressive_action(self, total_dices, last_bid):
        '''
        Always make the highest possible bid under estimate
        '''
        estimate = self.get_estimate_vector(total_dices)

        last_bid_quantity, last_bid_face = last_bid
        possible_actions = possible_actions_from_bid(total_dices, last_bid_quantity, last_bid_face)
        best_action = DOUBT_ACTION
        best_quantity = -1

        for action in possible_actions:
            bid_quantity, bid_face = action
            if action == DOUBT_ACTION:
                continue
            if bid_quantity <= estimate[bid_face - 1]:
                if bid_quantity > best_quantity:
                    best_quantity = bid_quantity
                    best_action = action

        return best_action

    def conservative_action(self, total_dices, last_bid):
        '''
        Always make the lowest possible bid under estimate
        '''
        estimate = self.get_estimate_vector(total_dices)

        last_bid_quantity, last_bid_face = last_bid
        possible_actions = possible_actions_from_bid(total_dices, last_bid_quantity, last_bid_face)
        best_action = DOUBT_ACTION
        best_quantity = total_dices + 1

        for action in possible_actions:
            bid_quantity, bid_face = action
            if action == DOUBT_ACTION:
                continue
            if bid_quantity <= estimate[bid_face - 1]:
                if bid_quantity < best_quantity:
                    best_quantity = bid_quantity
                    best_action = action

        return best_action

    def make_action(self, total_dices, last_bid):
        pass

class AggressiveRoboPlayer(RoboPlayer):
    def make_action(self, total_dices, last_bid):
        return self.aggressive_action(total_dices, last_bid)

class ConservativeRoboPlayer(RoboPlayer):
    def make_action(self, total_dices, last_bid):
        return self.conservative_action(total_dices, last_bid)
    
class HumanPlayer(Player):
    def make_action(self, total_dices, last_bid):
        print(f"Your dices: ")
        render_dices(self.dices)
        print(f"Total dices in game: {total_dices}")
        print(f"Last bid: {last_bid}")
        while True:
            print("Bid quantity: (-1 for doubt)")
            bid_quantity = int(input())
            if bid_quantity == -1:
                return DOUBT_ACTION
            print("Bid face (1-6): ")
            bid_face = int(input())
            if is_legal_bid(last_bid, (bid_quantity, bid_face)):
                return (bid_quantity, bid_face)
            print("Illegal bid, try again.")

class Game():
    STATS = {}
    def __init__(self, players, quiet=False):
        self.players = players
        self.n_players = len(players)
        self.start_idx = 0
        self.quiet = quiet
    
    def print(self, str_):
        if self.quiet:
            return
        print(str_)

    def play_round(self):
        self.print("---------------------")
        self.print("Starting a new round...")
        for p in self.players:
            p.new_round()
        
        total_dices = sum([p.n_dices for p in self.players])
        last_bid = (0, 0)
        current_player_idx = self.start_idx % self.n_players

        while True:

            current_player = self.players[current_player_idx]
            action = current_player.make_action(total_dices, last_bid)
            current_player.track_action(total_dices, last_bid, action)
            self.print(f"Player {current_player_idx}, {current_player.name} action: {action}")

            if action == DOUBT_ACTION:
                # Resolve doubt
                bidder_idx = (current_player_idx - 1) % self.n_players
                bidder = self.players[bidder_idx]
                actual_count = 0
                q, f = last_bid
                for p in self.players:
                    actual_count += p.dices[0]
                    if f != JOLLY_FACE:
                        actual_count += p.dices[f - 1]
                
                self.print(f"Actual count for face {f}: {actual_count}")

                if actual_count >= q:
                    # Bidder wins
                    self.start_idx = current_player_idx
                    current_player.lose_dice()
                    if (not current_player.is_alive()):
                        self.print(f"Player {current_player_idx} has been eliminated!")
                        self.players.pop(current_player_idx)
                        self.n_players -= 1
                else:
                    # Doubter wins
                    self.start_idx = bidder_idx
                    bidder.lose_dice()
                    if (not bidder.is_alive()):
                        self.print(f"Player {bidder_idx} has been eliminated!")
                        self.players.pop(bidder_idx)
                        self.n_players -= 1
                break
            else:
                last_bid = action
                current_player_idx = (current_player_idx + 1) % self.n_players
        
        if self.n_players == 1:
            winner = self.players[0]
            winner.win()
            self.print(f"We have a winner: {winner.name}")
            Game.STATS[winner.name] = Game.STATS.get(winner.name, 0) + 1
            return True
        
        return False

    def play_game(self, with_human=False):
        game_over = False
        while not game_over:
            game_over = self.play_round()
            if with_human:
                print("Press Enter to continue to next round...")
                input()

def train_against_random_models(n=10000, n_players=MAX_PLAYERS):
    models = [AggressiveRoboPlayer, ConservativeRoboPlayer, RLPlayer]
    others = random.choices(models, k=n_players - 1)
    return trainRL(n=n, opponent_models=others)

def trainRL(n=10000, n_players=MAX_PLAYERS, opponent_models=[AggressiveRoboPlayer, ConservativeRoboPlayer]):

    # Reset RLPlayer stats
    RLPlayer.GAMES = 0
    RLPlayer.WINS = 0
    
    for _ in range(n):
        rlplayer = RLPlayer("RL-Agent")
        rlplayer.track_stats()
        players = [rlplayer] + [m(f"Player-{i}") for i, m in enumerate(opponent_models[:n_players - 1])]
        game = Game(players, quiet=True)
        game.play_game()
    
    return RLPlayer.get_stats()



def main():
    print("Play against the trained RL agent!")
    
    game = Game([HumanPlayer("Human"), RLPlayer("RL1")])
    game.play_game(True)