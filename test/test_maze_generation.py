from random import Random

import networkx as nx
import numpy as np
import pytest

import pelita.maze_generator as mg
import pelita.layout as pl
import pelita.team as pt

SEED = 103525239

def layout_str_to_graph(l_str):
    l_dict = pl.parse_layout(l_str, strict=False)
    graph = pt.walls_to_graph(l_dict['walls'], shape=l_dict['shape'])
    graph.shape = l_dict['shape']
    return graph

maze_103525239 = """
################################
#.   .....#....##    .      . y#
# .# #########      # .    #  x#
#  # # .     .    # ######    ##
#  #    .    # ##.         #   #
#  ####### ###    #.      .# ..#
# .#     .  ..    ### # ##.#####
# ### ###### ####  .    #.   . #
# .   .#    .  #### ###### ### #
#####.## # ###    ..  .     #. #
#.. #.      .#    ### #######  #
#   #         .## #    .    #  #
##    ###### #    .     . # #  #
#a  #    . #      ######### #. #
#b .      .    ##....#.....   .#
################################
"""

def test_generate_maze_stability():
    # we only test that we keep in returning the same maze when the random
    # seed is fixed, in case something changes during future porting/refactoring
    new_layout = mg.generate_maze(rng=SEED)
    old_layout = pl.parse_layout(maze_103525239)
    assert old_layout == new_layout

def test_find_trapped_tiles():
    # This maze has one single chamber
    one_chamber = """############
                     #   #      #
                     #   #      #
                     # ###      #
                     #          #
                     #          #
                     ############"""

    graph = layout_str_to_graph(one_chamber)
    one_chamber_tiles, chambers = mg.find_trapped_tiles(graph, graph.shape[0], include_chambers=True)
    assert len(chambers) == 1
    tiles_1 = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2)}
    assert one_chamber_tiles == tiles_1


    two_chambers = """############
                      #   #      #
                      #   #      #
                      # ###  # ###
                      #      #   #
                      #      #   #
                      #      #   #
                      #      #   #
                      ############"""
    graph = layout_str_to_graph(two_chambers)
    two_chambers_tiles, chambers = mg.find_trapped_tiles(graph, graph.shape[0], include_chambers=True)
    assert len(chambers) == 2
    tiles_2 = {(8,3), (8,4), (8,5), (8,6), (8,7), (9,4), (9,5),
               (9,6), (9,7), (10, 4), (10,5), (10,6), (10, 7) }
    assert two_chambers_tiles == tiles_1 | tiles_2

    # only find chambers in the left side of the maze



@pytest.mark.xfail
def test_add_food():
    mini = """########
              #      #
              #      #
              ########"""
    maze = mg.str_to_maze(mini)
    # We have 12 empty squares in total, so 6 on the left side.
    # -> 2 slots are taken by the dividing border, so we have 4 left
    # -> 2 are reserved for pacmen, so we have only 2 slots for food
    # - check that we can indeed accommodate 2 pellets in the maze
    lmaze = maze.copy()
    mg.add_food(lmaze, 2)
    assert (lmaze == mg.F).sum() == 2
    # - check that if we add a wall in a free spot that is not reserved by the
    #   pacmen we can only accommodate 1 pellet
    lmaze = maze.copy()
    lmaze[1,2] = mg.W
    mg.add_food(lmaze, 1)
    assert (lmaze == mg.F).sum() == 1
    # - if we try to add more food, we complain
    lmaze = maze.copy()
    lmaze[1,2] = mg.W
    with pytest.raises(ValueError):
        mg.add_food(lmaze, 2)
    # - check that if no space is left we complain
    lmaze = maze.copy()
    lmaze[1,2], lmaze[2,2] = mg.W, mg.W
    with pytest.raises(ValueError):
        mg.add_food(lmaze, 1)
    # - check that we fail if we get passed unreasonable amounts of food
    lmaze = maze.copy()
    with pytest.raises(ValueError):
        mg.add_food(lmaze, -1)
    # - check that we can cope with no food to add
    lmaze = maze.copy()
    mg.add_food(lmaze, 0)
    assert np.all(lmaze == maze)


@pytest.mark.xfail
def test_distribute_food():
    maze_chamber = """############
                      #   #      #
                      #   #      #
                      # ###      #
                      #          #
                      #          #
                      ############"""

    maze = mg.str_to_maze(maze_chamber)
    graph = mg.walls_to_graph(maze)
    all_tiles = set(graph.nodes)
    width = maze.shape[1]
    chamber_tiles = mg.find_chambers(graph, width)

    # expected exceptions
    with pytest.raises(ValueError):
        mg.distribute_food(all_tiles, chamber_tiles, 0, len(all_tiles) + 1)

    with pytest.raises(ValueError):
        mg.distribute_food(all_tiles, chamber_tiles, 10, 8)

    # no intersection of food positions and chamber_tiles
    trapped_food = 0
    total_food = 10
    food = mg.distribute_food(all_tiles, chamber_tiles, trapped_food, total_food)
    assert len(set(food) & chamber_tiles) == trapped_food

    # trapped food is placed in chamber as requested
    # all food is placed as requested
    trapped_food = 3
    food = mg.distribute_food(all_tiles, chamber_tiles, trapped_food, total_food)
    assert len(food & chamber_tiles) == trapped_food
    assert len(food) == total_food

    # food is completely contained in chamber
    total_food = trapped_food = 3
    food = mg.distribute_food(all_tiles, chamber_tiles, trapped_food, total_food)
    assert food.issubset(chamber_tiles)
    assert len(food) == total_food

    # best effort placement of trapped food
    trapped_food = 10  # > 7 chamber_tiles
    total_food = 20
    food = mg.distribute_food(all_tiles, chamber_tiles, trapped_food, total_food)
    assert len(food & chamber_tiles) == len(chamber_tiles)
    assert len(food) == total_food

    # distribute only leftover food in chambers, fill non-chambers first
    trapped_food = 1
    free_tiles = all_tiles - chamber_tiles
    n_free_tiles = len(free_tiles)
    leftover_food = 2
    total_food = n_free_tiles + trapped_food + leftover_food
    food = mg.distribute_food(all_tiles, chamber_tiles, trapped_food, total_food)
    assert len(food) == total_food
    assert (food & free_tiles) == free_tiles
    assert len(food & chamber_tiles) == trapped_food + leftover_food

    # edge case, no food at all
    food = mg.distribute_food(all_tiles, chamber_tiles, 0, 0)
    assert len(food) == 0

np_maze_generator_86523 = """
################################
# #  ..  ..  # ##    # ..  #  y#
# # . .           .. ## ####  x#
#       .. . #     . #   . ## ##
#.#### #######    ####.#. .#   #
#. .  .        ## .# # #       #
#.#  ## ######     #  .### #####
#     ....   . ## ## #      #  #
#  #      # ## ## .   ....     #
##### ###.  #     ###### ##  #.#
#       # # #. ##        .  . .#
#   #. .#.####    ####### ####.#
## ## .   # .     # . ..       #
#a  #### ## ..           . . # #
#b  #  .. #    ## #  ..  ..  # #
################################
"""


@pytest.mark.parametrize('iteration', range(100))
def test_generate_maze(iteration):
    local_seed = SEED * iteration
    rng = Random(local_seed)

    # edge cases
    # width not even
    with pytest.raises(ValueError):
        mg.generate_maze(0, 0, 9, 10, rng=rng)

    # width too small
    with pytest.raises(ValueError):
        mg.generate_maze(0, 0, 2, 10, rng=rng)

    # height too small
    with pytest.raises(ValueError):
        mg.generate_maze(0, 0, 10, 2, rng=rng)


    width = rng.choice(range(16, 65, 2))
    height = rng.randint(8, 32)
    total_food = int(0.15 * width * height / 2)
    trapped_food = int(total_food / 3)

    ld = mg.generate_maze(trapped_food, total_food, width, height, rng=rng)

    walls = set(ld["walls"])

    border = set()
    for x in range(width):
        border.add((x, 0))
        border.add((x, height - 1))

    for y in range(height):
        border.add((0, y))
        border.add((width - 1, y))

    def split(nodes, width):
        left = [node for node in nodes if node[0] < width // 2]
        right = [node for node in nodes if node[0] >= width // 2]
        return left, right

    def is_full_mirror(left, right, width, height):
        right_copy = right[:]

        for x, y in left:
            mirrored = (width - 1 - x, height - 1 - y)
            if mirrored not in right:
                return False
            else:
                right_copy.remove(mirrored)

        if len(right_copy) > 0:
            return False

        return True

    # check that the maze is mirrored around the center
    left_walls, right_walls = split(ld["walls"], width)
    left_food, right_food = split(ld["food"], width)
    assert is_full_mirror(left_walls, right_walls, width, height)
    assert is_full_mirror(left_food, right_food, width, height)

    # check that the maze is completely enclosed by walls
    assert border.issubset(set(ld["walls"]))

    # check that there is enough food
    assert len(ld["food"]) == 2 * total_food

    # check that the bots are where they are supposed to be
    assert ld["bots"] == [
        (1, height - 3),
        (width - 2, 2),
        (1, height - 2),
        (width - 2, 1),
    ]

    # verify shape
    assert min(ld["walls"]) == (0, 0)
    assert max(ld["walls"]) == (width - 1, height - 1)

    # verify that the types in the dictionary are as expected
    def is_seq_of_tuples(seq_type, thing):
        assert type(thing) is seq_type
        for item in thing:
            assert type(item) is tuple

    is_seq_of_tuples(tuple, ld["walls"])
    is_seq_of_tuples(list, ld["food"])
    is_seq_of_tuples(list, ld["bots"])
    assert type(ld["shape"]) is tuple

    # verify that we generate exactly the same maze if started with the same seed
    seed = rng.randint(1,100000)
    l1 = mg.generate_maze(trapped_food, total_food, width, height, rng=seed)
    l2 = mg.generate_maze(trapped_food, total_food, width, height, rng=seed)
    assert l1 == l2

@pytest.mark.parametrize('iteration', range(100))
def test_generate_maze_food(iteration):
    local_seed = SEED + iteration
    rng = Random(local_seed)

    width = 10
    height = 5
    total_food = 10
    trapped_food = 0
    ld = mg.generate_maze(trapped_food, total_food, width, height, rng=rng)
    # check that we never place food on the border
    x_food = {x for (x, y) in ld['food']}
    assert width//2 not in x_food
    assert width//2 - 1 not in x_food
