import random

import matplotlib
import numpy as np


import socket
import threading
import ast
matplotlib.use('Agg')  # Use a non-interactive backend


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
    def __init__(self, host='localhost', port=12345,):
        self.host = host
        self.port = port
        self.server_socket = None

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
        minion_id = None
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

                if data.startswith("your_turn:"):
                    minion_id = int(data.split(":")[1])
                    if not move_ids:
                        continue
                    move_id = random.choice(move_ids)
                    selected_move = move_ids[move_id]
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
                        selected_position = 1
                        valid_positions.remove(selected_position)
                        if i == target_amount - 1:
                            targets += str(selected_position)
                        else:
                            targets += str(selected_position) + "|"
                    print(f"Selected targets: {targets}")
                    self.send_data(client_socket, targets)

                elif data.startswith("turn_ended:"):
                    turn_count += 1

                elif data.startswith("battle ended"):
                    outcome = data.split(":")[1].strip()

        except ConnectionResetError:
            pass
        finally:
            client_socket.close()
            self.server_socket.close()

    @staticmethod
    def send_data(client_socket, message):
        try:
            client_socket.sendall(message.encode('utf-8'))
            print(f"Sent data: {message}")
        except Exception as e:
            print(f"Failed to send data: {e}")


if __name__ == "__main__":
    server = GameSocketServer()
    server.start_server()
