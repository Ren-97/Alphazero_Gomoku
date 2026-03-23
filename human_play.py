"""
human VS mcts_alphaZero
"""

import logging
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet
from az_logging import setup_logging


logger = logging.getLogger(__name__)


class Human(object):

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        while True:
            try:
                location_str = input("Your move: ")
                location = [int(n, 10) for n in location_str.split(",")]
                move = board.location_to_move(location)
            except Exception:
                move = -1

            if move != -1 and move in board.availables:
                return move
            logger.warning("Invalid move.")

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 4
    width, height = 6, 6
    model_file = 'best_policy_6_6_4.pth'
    try:
        setup_logging(log_file=None)
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=False)
        mcts_player = MCTSPlayer(
            best_policy.policy_value_fn,
            c_puct=5,
            n_playout=400,  # set larger n_playout for better performance
        )

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        logger.info("quit")


if __name__ == '__main__':
    run()
