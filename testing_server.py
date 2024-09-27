import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import json
import socket
import threading
import sys
import time
import ast


# Define the neural network for the Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_size, action_size, max_targets):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size + max_targets)
        self.max_targets = max_targets

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Define the agent
class Agent:
    def __init__(self, state_size, action_size, max_targets, seed, model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.max_targets = max_targets
        self.seed = random.seed(seed)

        self.qnetwork_local = DQN(state_size, action_size, max_targets)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            raise ValueError(f"Model file {model_path} not found")

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()  # Set to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        action_size = self.action_size
        max_targets = self.max_targets

        move_action = np.argmax(action_values.cpu().data.numpy()[:, :action_size])
        target_actions = np.argsort(-action_values.cpu().data.numpy()[:, action_size:action_size + max_targets])[0]

        # Ensure move_action is within the valid range
        move_action = max(0, min(move_action, action_size - 1))

        return move_action, target_actions

    def load_model(self, filepath):
        self.qnetwork_local.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")


# Additional functions for the game loop
def pad_list(lst, length, pad_value=0):
    return lst + [pad_value] * (length - len(lst))


def get_state(minions, moves, minion_id, max_minions=10, max_moves=8):
    state = []

    for minion in minions:
        if minion['minion_id'] == minion_id:
            state.extend([
                minion['minionDexID'],
                minion['currHealth'],
                minion['currEnergy'],
                minion['currShield'],
                minion['m_currExhaust'],
                minion['m_moveOrderPosition']
            ])

    ally_minions = [m for m in minions if m['side'] == 'ally']
    enemy_minions = [m for m in minions if m['side'] == 'enemy']

    for minion in pad_list(ally_minions, max_minions // 2):
        state.extend([
            minion['minionDexID'] if minion else 0,
            minion['currHealth'] if minion else 0,
            minion['currEnergy'] if minion else 0,
            minion['currShield'] if minion else 0,
            minion['m_currExhaust'] if minion else 0,
            minion['m_moveOrderPosition'] if minion else 0,
            1 if minion else 0  # Indicator for ally
        ])

    for minion in pad_list(enemy_minions, max_minions // 2):
        state.extend([
            minion['minionDexID'] if minion else 0,
            minion['currHealth'] if minion else 0,
            minion['currEnergy'] if minion else 0,
            minion['currShield'] if minion else 0,
            minion['m_currExhaust'] if minion else 0,
            minion['m_moveOrderPosition'] if minion else 0,
            0 if minion else 0  # Indicator for enemy
        ])

    for move in pad_list(moves, max_moves):
        state.extend([
            move['m_moveID'] if move else 0,
            move['m_damage'] if move else 0,
            move['m_energyUsed'] if move else 0,
            move['m_enemiesItHits'] if move else 0,
            move['m_additionalRandomDamage'] if move else 0
        ])

    return np.array(state)


def get_reward(minions, prev_minions, outcome):
    reward = 0
    if outcome == "victory":
        reward += 10.0
    elif outcome == "defeat":
        reward -= 10.0
    try:
        for minion, prev_minion in zip(minions, prev_minions):
            if minion['side'] == 'ally':
                if minion['currHealth'] <= 0 and prev_minion['currHealth'] > 0:
                    reward -= 2.0
                elif 0 < minion['currHealth'] < prev_minion['currHealth']:
                    reward -= (prev_minion['currHealth'] - minion['currHealth']) / minion['currHealthStat']
            elif minion['side'] == 'enemy':
                if minion['currHealth'] <= 0 and prev_minion['currHealth'] > 0:
                    reward += 2.0
                elif 0 < minion['currHealth'] < prev_minion['currHealth']:
                    reward += (prev_minion['currHealth'] - minion['currHealth']) / minion['currHealthStat']
    except Exception as e:
        print("reward error:", e)
    return reward


def is_ally(minion_id, minions):
    for minion in minions:
        if minion["minion_id"] == minion_id:
            return minion["side"] == "ally"
    return False


def select_target_positions(minions, is_ally_turn):
    valid_positions = []
    for minion in minions:
        if minion["currHealth"] > 0:
            if (is_ally_turn and minion["side"] == "enemy") or (not is_ally_turn and minion["side"] == "ally"):
                valid_positions.append(minion["position"])
    return valid_positions


class GameSocketServer:
    def __init__(self, host='localhost', port=12345, model_path=None):
        self.host = host
        self.port = port
        self.agent = Agent(state_size=116, action_size=3, max_targets=4,
                           seed=0, model_path=model_path)  # Adjust state_size and action_size as needed
        self.epsilon = 0.0  # Disable exploration

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print("Game server started, waiting for connections...")

        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        move_ids = None
        minions = []
        moves = None
        state = None
        next_state = None
        prev_minions = None
        minion_id = None
        action = None
        turn_count = 0
        try:
            while True:
                data = str(client_socket.recv(4096).decode('utf-8'))
                if not data:
                    break

                if data.startswith("move:"):
                    moves = []
                    move_entries = data.strip().split("\n")
                    for entry in move_entries:
                        move_data = entry.split("move:")[1]
                        try:
                            move = ast.literal_eval(move_data)
                        except SyntaxError:
                            move_data, your_turn_data = move_data.split("your_turn:")
                            move = ast.literal_eval(move_data.strip())
                            data = "your_turn:" + your_turn_data.strip()
                        moves.append(move)
                    move_ids = [move['m_moveID'] for move in moves]

                if data.startswith("minion:"):
                    minions = []
                    minion_entries = data.strip().split("\n")
                    for entry in minion_entries:
                        try:
                            minion_data = entry.split("minion:")[1]
                        except Exception as e:
                            print(data)
                            raise e
                        minion = ast.literal_eval(minion_data)

                        minions.append(minion)
                    if prev_minions is None:
                        prev_minions = [minion.copy() for minion in minions]

                if data.startswith("your_turn:"):
                    minion_id = int(data.split(":")[1])
                    if not move_ids:
                        continue
                    state = get_state(minions, moves, minion_id)
                    action, target_actions = self.agent.act(state, self.epsilon)
                    while action >= len(move_ids):
                        print(f"Action {action} is out of bounds for move_ids of length {len(move_ids)}")
                        action = random.choice(np.arange(len(move_ids)))  # Fallback to random valid action
                        print("random action :", action)
                    selected_move = move_ids[action]
                    print("action :", action)
                    print(f"Selected move: {selected_move}")
                    if is_ally(minion_id, minions):
                        self.send_data(client_socket, f"allymove:{selected_move}")
                    else:
                        self.send_data(client_socket, f"enemymove:{selected_move}")

                elif data.startswith("select_targets:"):
                    target_amount = int(data.split(":")[1])
                    valid_positions = select_target_positions(minions, is_ally(minion_id, minions))
                    targets = "target:"
                    for i in range(target_amount):
                        if not valid_positions:
                            break
                        selected_position = valid_positions[int(target_actions[i] % len(valid_positions))]
                        valid_positions.remove(selected_position)
                        if i == target_amount - 1:
                            targets += str(selected_position)
                        else:
                            targets += str(selected_position) + "|"
                    print(f"Selected targets: {targets}")
                    self.send_data(client_socket, targets)

                elif data.startswith("turn_ended:"):
                    turn_count += 1
                    next_state = get_state(minions, moves, minion_id)
                    prev_minions = [minion.copy() for minion in minions]

                elif data.startswith("battle ended"):
                    outcome = data.split(":")[1].strip()
                    if action is not None and turn_count > 0:
                        reward = get_reward(minions, prev_minions, outcome)
                        print(f"Battle ended. Outcome: {outcome}. Reward: {reward}")


        except ConnectionResetError:
            pass
        finally:
            client_socket.close()
            self.server_socket.close()

    def send_data(self, client_socket, message):
        try:
            client_socket.sendall(message.encode('utf-8'))
            print(f"Sent data: {message}")
        except Exception as e:
            print(f"Failed to send data: {e}")

if __name__ == "__main__":
    server = GameSocketServer(model_path="models/model_battle_1919.pth")
    server.start_server()
