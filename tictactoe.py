import gym
from gym import spaces
import numpy as np
import torch
import random

class TicTacToe(gym.Env):
    def __init__(self):
        super(TicTacToe, self).__init__()
        
        # Define action space: 9 slots where to place the next marker
        self.action_space = spaces.Discrete(9)
        
        # Observation space is a 3x3 board where each cell can be:
        # 0: empty, 1: player X, 2: player O
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int8)
        
        # Initialize state
        self.reset()

    def step(self, action):
        # Check if action is valid
        if self.done:
            raise ValueError("Game is done, call reset() first")
        
        # Action is 0-8 and represents the cell to place the marker
        row = action // 3
        col = action % 3
        
        # Check if the selected position is empty
        if self.state[row, col] != 0:
            return self.state, -10, True, False, {"invalid_move": True}
        
        # Make move
        self.state[row, col] = self.current_player
        
        # Check for winner
        winner = self._check_winner()
        if winner:
            reward = 1 if winner == self.current_player else -1
            self.done = True
            return self.state, reward, True, False, {"winner": winner}
        
        # Check for draw
        if torch.all(self.state != 0):
            self.done = True
            return self.state, 0, True, False, {"draw": True}
        
        # Switch player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2
        
        return self.state, 0, False, False, {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Clear the board
        self.state = torch.zeros((3, 3), dtype=torch.int8)
        self.current_player = 1  # Player X starts
        self.done = False
        return self.state, {}  # Return state and info dict
    
    def _check_winner(self):
        # Check rows, columns and diagonals
        for player in [1, 2]:
            # Check rows and columns
            for i in range(3):
                if torch.all(self.state[i, :] == player) or torch.all(self.state[:, i] == player):
                    return player
            
            # Check diagonals
            if torch.all(torch.diag(self.state) == player) or torch.all(torch.diag(torch.fliplr(self.state)) == player):
                return player
        
        return None

    def render(self):
        # Simple console rendering
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for row in self.state:
            print('|', end='')
            for cell in row:
                print(f' {symbols[cell.item()]} |', end='')
            print('\n-----------')


# A class representing a node of in the state graph diagram
# Each state represents a certain board configuration
# p_win_x is the probability that x wins from that board
# p_win_o is the probabily that o wins from that board

class Node():
    def __hash__(self):
        return hash(self.board)
    
    def __init__(self, board_matrix=None):
        if board_matrix != None:
            self.board = board_matrix.clone()
        else:
            self.board = torch.zeros((3, 3), dtype=torch.int8) # This is the actual board matrix
    
        self.p_win_x_ = 0.5

        if self._check_winner() == 1:
            self.p_win_x_ = 1
        
        if self._check_winner() == 2:
            self.p_win_x_ = 0
            
    # Actual "value" of this state
    def p_win(self, player):
        if player == 1:
            return self.p_win_x_
        else:
            return 1 - self.p_win_x_
    
    def set_value(self, value, player):
        if player == 1:
            self.p_win_x_ = value
        else:
            self.p_win_x_ = 1 - value
    
    def p_win_x(self):
        return self.p_win(1)
    
    def p_win_o(self):
        return self.p_win(2)

    def clone_board(self):
        new_node = Node(self.board)
        return new_node

    def next_player(self):
        # Just count the number of filled cells
        # If it's even, next player is X (first to play)
        # If it's odd, next player is O
        count = torch.sum(self.board != 0).item()
        if count % 2 == 0:
            return 1 # X
        return 2 # O
        
    def _check_winner(self):
        # Check rows, columns and diagonals
        for player in [1, 2]:
            # Check rows and columns
            for i in range(3):
                if torch.all(self.board[i, :] == player) or torch.all(self.board[:, i] == player):
                    return player
            
            # Check diagonals
            if torch.all(torch.diag(self.board) == player) or torch.all(torch.diag(torch.fliplr(self.board)) == player):
                return player
            
            return None

class Player():
    # By default, player 1 is the first to play (X)
    def __init__(self, player_number = 1):
        self.nodes = {} # Cache the boards that I've seen so far
        self.player = player_number
        self.new_game()

    def add_node(self, node):
        key = tuple(i.item() for i in node.board.flatten())
        self.nodes[key] = node
    
    def get_node(self, board_matrix):
        key = tuple(i.item() for i in board_matrix.flatten())
        if not (key in self.nodes):
            self.add_node(Node(board_matrix))
        
        return self.nodes[key]
    
    def new_game(self, player_number = 1):
        self.player = player_number
        self.moves = 0
        self.board_history = []

    def backpropagate(self, reward):
        
        step = 0.1
        last_board = self.board_history[-1]
        last_node = self.get_node(last_board)
        old_value = last_node.p_win(self.player)
        new_value = old_value + step*(reward - old_value)
        last_node.set_value(new_value, self.player)

        for i in range(len(self.board_history) - 2, -1, -1):
            next_board = self.board_history[i + 1]
            next_node = self.get_node(next_board)
            cur_board = self.board_history[i]
            # TODO: also apply the learning for the rotated and flipped boards
            cur_node = self.get_node(cur_board)
            old_value = cur_node.p_win(self.player)
            new_value = old_value + step*(next_node.p_win(self.player) - old_value)
            cur_node.set_value(new_value, self.player)
    
    def pick_action(self, board):
        possible_boards = [None] * 9
        action_probs = [0] * 9
        
        for i in enumerate(board.flatten() == 0):
            if not i[1]:
                continue
            
            action = i[0]
            # Action is 0-8 and represents the cell to place the marker
            row = action // 3
            col = action % 3
            new_board = board.clone()
            new_board[row, col] = self.player
            possible_boards[action] = self.get_node(new_board)
            action_probs[action] = self.get_node(new_board).p_win(self.player)

        # Get the action with the highest probability
        best_action = max(range(len(action_probs)), key=lambda i: action_probs[i])
        if random.random() < 0.1:
            # Pick a random valid action (where possible_boards is not None)
            valid_actions = [i for i, b in enumerate(possible_boards) if b is not None]
            if valid_actions:
                best_action = random.choice(valid_actions)
    
        self.moves += 1
        self.board_history.append(possible_boards[best_action].board)

        return best_action

    def visualize_rewards(self):
        rewards = {n: self.nodes[n].p_win(self.player) for n in self.nodes}
        return rewards

player1 = Player()
player2 = Player()

def train(n_games=100):
    env = TicTacToe()
    obs, _ = env.reset()

    # Example game loop
    play_again = True
    played = 0
    while play_again:
        played +=1 
        obs, _ = env.reset()
        player1.new_game(player_number=1)
        player2.new_game(player_number=2)

        done = False
        while not done:


            # Replace with your agent's action selection
            obs, reward, done, _, info = env.step(player1.pick_action(env.state))

            if done:
                break

            obs, reward, done, _, info = env.step(player2.pick_action(env.state))

            env.render()
            
        env.render()


        if "winner" in info:
            print(f"Player {info['winner']} wins!")
            if info['winner'] == 1:
                player1.backpropagate(1)
                player2.backpropagate(0)

            if info['winner'] == 2:
                player1.backpropagate(0)
                player2.backpropagate(1)
        elif "draw" in info:
            print("Game is a draw!")
            player1.backpropagate(0)
            player2.backpropagate(0)
        

        play_again = played < n_games or input("Play again? (y/n): ") == "y"

def play(n_games=1):
    env = TicTacToe()
    obs, _ = env.reset()

    # Example game loop
    play_again = True
    played = 0
    while play_again:
        played +=1 
        obs, _ = env.reset()
        player1.new_game(player_number=1)

        done = False
        while not done:


            # Replace with your agent's action selection
            obs, reward, done, _, info = env.step(player1.pick_action(env.state))

            if done:
                break
            env.render()
            
            my_action = int(input("Select a move 0-8:"))
            obs, reward, done, _, info = env.step(my_action)

            env.render()
            
        env.render()


        if "winner" in info:
            print(f"Player {info['winner']} wins!")
            if info['winner'] == 1:
                player1.backpropagate(1)

            if info['winner'] == 2:
                player1.backpropagate(0)
        elif "draw" in info:
            print("Game is a draw!")
            player1.backpropagate(0)
        

        play_again = played < n_games or input("Play again? (y/n): ") == "y"


if __name__ == "__main__":
    train(1000)
    print(player1.visualize_rewards())

