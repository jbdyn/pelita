import pytest

import subprocess
import sys
import tempfile

from pelita.simplesetup import ZMQClientError
from pelita import libpelita


class TestLibpelitaUtils:
    def test_firstNN(self):
        assert libpelita.firstNN(None, False, True) == False
        assert libpelita.firstNN(True, False, True) == True
        assert libpelita.firstNN(None, None, True) == True
        assert libpelita.firstNN(None, 2, True) == 2
        assert libpelita.firstNN(None, None, None) == None
        assert libpelita.firstNN() == None

def test_call_pelita():
    rounds = 200
    viewer = 'ascii'
    filter = 'small'

    teams = ["pelita/player/StoppingPlayer", "pelita/player/StoppingPlayer"]
    (state, stdout, stderr) = libpelita.call_pelita(teams, rounds=rounds, viewer='null', filter=filter, seed=None)
    assert state['gameover'] is True
    assert state['whowins'] == 2
    # Quick assert that there is text in stdout
    assert len(stdout.split('\n')) == 6

    teams = ["pelita/player/SmartEatingPlayer", "pelita/player/StoppingPlayer"]
    (state, stdout, stderr) = libpelita.call_pelita(teams, rounds=rounds, viewer=viewer, filter=filter, seed=None)
    assert state['gameover'] is True
    assert state['whowins'] == 0

    teams = ["pelita/player/StoppingPlayer", "pelita/player/SmartEatingPlayer"]
    (state, stdout, stderr) = libpelita.call_pelita(teams, rounds=rounds, viewer=viewer, filter=filter, seed=None)
    assert state['gameover'] is True
    assert state['whowins'] == 1


def test_check_team_external():
    assert libpelita.check_team("pelita/player/StoppingPlayer") == "Stopping"

def test_check_team_external_fails():
    with pytest.raises(ZMQClientError):
        libpelita.check_team("Unknown Module")

def test_check_team_internal():
    def move(b, s):
        return b.position, s
    assert libpelita.check_team(move) == "local-team (move)"


def test_write_replay_is_idempotent():
    # TODO: The replay functionality could be added to call_pelita
    # so we don’t have to run the subprocess ourselves
    with tempfile.NamedTemporaryFile() as f:
        with tempfile.NamedTemporaryFile() as g:
            # run a quick game and save the game states to f

            cmd = [libpelita.get_python_process(), '-m', 'pelita.scripts.pelita_main',
                    '--write-replay', f.name,
                    '--filter', 'small',
                    '--null']

            subprocess.check_output(cmd)

            f.seek(0)
            first_run = f.read()
            # check that we received something
            assert len(first_run) > 0

            # run a replay of f and store in g
            cmd = [libpelita.get_python_process(), '-m', 'pelita.scripts.pelita_main',
                    '--write-replay', g.name,
                    '--replay', f.name,
                    '--null']

            subprocess.check_output(cmd)

            g.seek(0)
            second_run = g.read()
            # check that f and g have the same content
            assert first_run == second_run

