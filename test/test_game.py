import glob
import pytest
import numpy as np
from pelita.game import initial_positions
from pelita.game import play_turn

from pelita import layout

def test_initial_positions_basic():
    walls = [(0, 0),
             (0, 1),
             (0, 2),
             (0, 3),
             (1, 0),
             (1, 3),
             (2, 0),
             (2, 1),
             (2, 3),
             (3, 0),
             (3, 1),
             (3, 3),
             (4, 0),
             (4, 1),
             (4, 3),
             (5, 0),
             (5, 3),
             (6, 0),
             (6, 3),
             (7, 0),
             (7, 1),
             (7, 2),
             (7, 3)]
    out = initial_positions(walls)
    exp = [(1, 1), (6, 2), (1, 2), (6, 1)]
    assert len(out) == 4
    assert out == exp

@pytest.mark.parametrize('n_times', range(30))
def test_initial_positions_same_in_layout_random(n_times):
    seedval = np.random.randint(2**32)
    print(f'seedval: {seedval}')
    # for i in range(n_times):
    l = layout.get_random_layout()
    parsed_l = layout.parse_layout(l[1])
    exp = parsed_l["bots"]
    walls = parsed_l["walls"]
    out = initial_positions(walls)
    assert out == exp

@pytest.mark.parametrize('layout_name', layout.get_available_layouts())
def test_initial_positions_same_in_layout(layout_name):
    l = layout.load_layout(layout_name=layout_name)
    parsed_l = layout.parse_layout(l[1])
    exp = parsed_l["bots"]
    walls = parsed_l["walls"]
    out = initial_positions(walls)
    assert out == exp


def test_play_turn():
    seedval = np.random.randint(2**32)
    print(f'seedval: {seedval}')
    l = layout.get_random_layout()
    parsed_l = layout.parse_layout(l[1])
    game_state = {
                  "food": parsed_l["food"],
                  "walls": parsed_l["walls"],
                  "bots": parsed_l["bots"],
                  "max_round": 300,
                  "team_names": ("a", "b"),
                  "turn": 1,
                  "round": 1,
                  "timeout": [],
                  "gameover": False,
                  "whowins": None,
                  "team_say": "bla",
                  "score": 0,
                  "deaths": 0,
                  }
    game_state = play_turn(game_state, 0, (1, 2))
    # TODO is move legal?
