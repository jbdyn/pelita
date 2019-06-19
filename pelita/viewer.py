""" The observers. """

import json
import logging

import zmq

from . import layout

_logger = logging.getLogger(__name__)



class ProgressViewer:
    def show_state(self, game_state):
        score = game_state["score"]
        round_index = game_state["round"]
        if round_index is None:
            return
        game_time = game_state["max_rounds"]
        percentage = int(100.0 * round_index / game_time)
        if game_state["turn"] is not None:
            if game_state["turn"] % 2 == 0:
                bot_sign = f'\033[94m{game_state["turn"]}\033[0m'
            elif game_state["turn"] % 2 == 1:
                bot_sign = f'\033[91m{game_state["turn"]}\033[0m'
        else:
            bot_sign = ' '
        string = ("[%s] %3i%% (%i / %i) [%s]" % (
                    bot_sign, percentage,
                    round_index, game_time,
                    ":".join(str(s) for s in score)))
        print(string + ("\b" * len(string)), flush=True)

        if game_state["gameover"]:
            state = {}
            state.update(game_state)
            del state['walls']
            del state['food']

            print()
            print("Final state:", state)

class AsciiViewer:
    """ A viewer that dumps ASCII charts on stdout. """

    def show_state(self, game_state):
        uni_str = layout.layout_as_str(walls=game_state['walls'],
                                       food=game_state['food'],
                                       bots=game_state['bots'])

        # Everything that we print explicitly is removed from the state dict.
        state = {}
        state.update(game_state)
        del state['walls']
        del state['food']
        del state['bots']
        del state['round']
        del state['turn']
        del state['score']

        info = (
            "Round: {round!r} Turn: {turn!r} Score {s0}:{s1}\n"
            "Game State: {state!r}\n"
            "\n"
            "{universe}"
        ).format(round=game_state["round"],
                 turn=game_state["turn"],
                 s0=game_state["score"][0],
                 s1=game_state["score"][1],
                 state=state,
                 universe=uni_str)

        print(info)
        if state.get("gameover"):
            if state["whowins"] == 2:
                print("Game Over: Draw.")
            else:
                winner = game_state["team_names"][state["whowins"]]
                print(f"Game Over: Team: '{winner}' wins!")


class ReplyToViewer:
    """ A viewer which dumps to a given stream.
    """
    def __init__(self, reply_to):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PAIR)

        # Wait max linger ms for a socket to connect
        # before giving up.
        self.sock.linger = 1000

        self.sock.connect(reply_to)
        _logger.debug(f"Connecting zmq.PAIR to {reply_to}")

        self.pollout = zmq.Poller()
        self.pollout.register(self.sock, zmq.POLLOUT)

    def _send(self, message):
        socks = dict(self.pollout.poll(300))
        if socks.get(self.sock) == zmq.POLLOUT:
            as_json = json.dumps(message)
            self.sock.send_unicode(as_json, flags=zmq.NOBLOCK)

    def show_state(self, game_state):
        self._send(game_state)


class DumpingViewer:
    """ A viewer which dumps to a given stream.
    """
    def __init__(self, stream):
        self.stream = stream

    def _send(self, message):
        as_json = json.dumps(message)
        self.stream.write(as_json)
        # We use 0x04 (EOT) as a separator between the events.
        # The additional newline is for improved readability
        # and should be ignored by the Python json reader.
        self.stream.write("\x04\n")
        self.stream.flush()

    def show_state(self, game_state):
        self._send(game_state)


class ResultPrinter:
    def show_state(self, state):
        if state["gameover"]:
            self.print_possible_winner(state)

    def print_possible_winner(self, state):
        """ Checks the game state for a winner.

        This is needed for pelita.scripts parsing the output.
        """
        winning_team = state.get("whowins")
        if winning_team in (0, 1):
            winner = state['team_names'][winning_team]
            loser = state['team_names'][1 - winning_team]
            winner_score = state['score'][winning_team]
            loser_score = state['score'][1 - winning_team]
            msg = f"Finished. '{winner}' won over '{loser}'. ({winner_score}:{loser_score})"
        elif winning_team == 2:
            t1, t2 = state['team_names']
            s1, s2 = state['score']
            msg = f"Finished. '{t1}' and '{t2}' had a draw. ({s1}:{s2})"
        else:
            return

        # We must flush, else our forceful stopping of Tk
        # won't let us pipe it.
        print(msg, flush=True)
