"""
Generate mazes for pelita

The algorithm is a form of binary space partitioning:

* start with an empty grid enclosed by walls
* draw a wall with k gaps, dividing the grid in 2 partitions
* repeat recursively for each sub-partitions, where walls have k/2 gaps
* pacmen are always in the bottom left and in the top right
* Food is distributed according to specifications

Notes:
The final maze includes is created by first generating the left half maze and then
generating the right half by making mirroring the left half. The resulting maze is
centrosymmetric.

Definitions:
Articulation point - a node in the graph that if removed would disconnect the
    graph in to two subgraphs. Find them with: nx.articulation_points(graph)

Dead-end - nodes in the graph with connectivity 1 -> that node is necessarily
    connected to an articulation point. Find them with
    {node for node in graph if graph.degree(node) == 1}

Tunnel: nodes in a sequence of articulation points connecting with a dead-end

Chamber: nodes connected to the main graph by a single articulation point -> it
    is basically a space with an entrance of a single node


Inspired by code by Dan Gillick
Completely rewritten by Pietro Berkes
Rewritten again (but not completely) by Tiziano Zito
Rewritten completely by Jakob Zahn & Tiziano Zito
"""

import networkx as nx

from .base_utils import default_rng
from .team import walls_to_graph


def mirror(nodes, width, height):
    nodes = set(nodes)
    other = set((width - 1 - x, height - 1 - y) for x, y in nodes)
    return nodes | other


def sample_nodes(nodes, k, rng=None):
    rng = default_rng(rng)

    if k < len(nodes):
        return set(rng.sample(sorted(nodes), k=k))
    else:
        return nodes


def find_trapped_tiles(graph, width, include_chambers=False):
    main_chamber = set()
    chamber_tiles = set()

    for chamber in nx.biconnected_components(graph):
        max_x = max(chamber, key=lambda n: n[0])[0]
        min_x = min(chamber, key=lambda n: n[0])[0]
        if min_x < width // 2 <= max_x:
            # only the main chamber covers both sides
            # our own mazes should only have one central chamber
            # but other configurations could have more than one
            main_chamber.update(chamber)
            continue
        else:
            chamber_tiles.update(set(chamber))

    # remove shared articulation points with the main chamber
    chamber_tiles -= main_chamber

    # combine connected subgraphs
    if include_chambers:
        subgraphs = graph.subgraph(chamber_tiles)
        chambers = list(nx.connected_components(subgraphs))
    else:
        chambers = []

    return chamber_tiles, chambers


def distribute_food(all_tiles, chamber_tiles, trapped_food, total_food, rng=None):
    rng = default_rng(rng)

    if trapped_food > total_food:
        raise ValueError(
            f"number of trapped food ({trapped_food}) must not exceed total number of food ({total_food})"
        )

    if total_food > len(all_tiles):
        raise ValueError(
            f"number of total food ({total_food}) exceeds available tiles in maze ({len(all_tiles)})"
        )

    free_tiles = all_tiles - chamber_tiles

    # distribute as much trapped food in chambers as possible
    tf_pos = sample_nodes(chamber_tiles, trapped_food, rng=rng)

    # distribute remaining food outside of chambers
    free_food = total_food - len(tf_pos)

    ff_pos = sample_nodes(free_tiles, free_food, rng=rng)

    # there were not enough tiles to distribute all leftover food
    leftover_food = total_food - len(ff_pos) - len(tf_pos)
    if leftover_food > 0:
        leftover_tiles = chamber_tiles - tf_pos
        leftover_food_pos = sample_nodes(leftover_tiles, leftover_food, rng=rng)
    else:
        leftover_food_pos = set()

    return tf_pos | ff_pos | leftover_food_pos


class Partition:
    # partition for binary space partitioning

    def __init__(self, pmin, pmax, vertical, rng=None):
        # minimum and maximum coordinates
        self.xmin, self.ymin = pmin
        self.xmax, self.ymax = pmax

        # wall orientation
        self.vertical = vertical

        # random number generator for later use in splitting
        self.rng = default_rng(rng)

        # choose wall position range depending on its orientation
        pos_min, pos_max = (
            (self.xmin, self.xmax) if vertical else
            (self.ymin, self.ymax)
        )

        # the offset from the left and right partition wall
        padding = 2

        # sample wall position
        self.pos = self.rng.randint(pos_min + padding, pos_max - padding)

    @property
    def pmin(self):
        # top left partition point
        return (self.xmin, self.ymin)

    @property
    def pmax(self):
        # bottom right partition point
        return (self.xmax, self.ymax)

    @property
    def wmin(self):
        # top/left wall point
        return (
            (self.pos, self.ymin) if self.vertical else
            (self.xmin, self.pos)
        )

    @property
    def wmax(self):
        # bottom/right wall point
        return (
            (self.pos, self.ymax) if self.vertical else
            (self.xmax, self.pos)
        )

    @property
    def width(self):
        # partition width
        return self.xmax - self.xmin + 1

    @property
    def height(self):
        # partition height
        return self.ymax - self.ymin + 1

    @property
    def length(self):
        # splittable length
        return self.width if self.vertical else self.height

    @property
    def wall(self):
        # all wall points
        if self.vertical:
            return [(self.pos, y) for y in range(self.ymin, self.ymax + 1)]
        else:
            return [(x, self.pos) for x in range(self.xmin, self.xmax + 1)]

    def split(self):
        # split the current partition into two inscribed ones
        children = list()
        cls = type(self)

        for pmin, pmax in (
            # top/left child partition
            (self.pmin, self.wmax),
            # bottom/right child partition
            (self.wmin, self.pmax),
        ):
            try:
                # rotate wall orientation for the next partition level
                child = cls(pmin, pmax, not self.vertical, rng=self.rng)
            except ValueError:
                # the wall position range is illegal
                pass
            else:
                # the child partition initialized well, so add it
                children.append(child)

        return children


def sample_wall(wall, walls, ngaps, rng=None):
    # choose wall points with respect to partition entrances

    rng = default_rng(rng)

    # avoid blocked entrances by removing additional wall points if needed
    start = 2 if wall[0] not in walls else 1
    end = -2 if wall[-1] not in walls else -1

    wall = wall[start:end]

    # cap the sample size accordingly
    return rng.sample(wall, k=max(0, len(wall) - ngaps))


def add_walls(partition, walls, ngaps, rng=None):
    # use binary space partitioning

    rng = default_rng(rng)

    # copy to avoid side effects
    walls = walls.copy()

    # store partitions in an expanding list with the number of gaps in the wall
    partitions = [(partition, ngaps)]

    # The loop is always exiting, since partitions always shrink,
    # no new partitions are added once they shrank below a threshold and
    # the loop draines the list in every iteration
    while len(partitions) > 0:
        # get the next partition
        partition, ngaps = partitions.pop()

        # adjust the padding around a partition's inner wall;
        # the higher the upper bound, the more spacious the chambers are
        if partition.length < rng.randrange(5, 7):
            continue

        # split the current partition
        children = partition.split()

        # add children with the adjusted number of wall gaps to the buffer;
        # the higher the lower gap bound, the more are the walls detached
        partitions.extend((child, max(1, ngaps // 2)) for child in children)

        # collect the current partition's wall into the global wall set
        wall = sample_wall(partition.wall, walls, ngaps, rng=rng)
        walls |= set(wall)

    return walls


def generate_half_maze(width, height, ngaps_center, bots_pos, rng=None):
    rng = default_rng(rng)

    # outer walls of the full maze: top, bottom, left and right
    walls = {(x, 0) for x in range(width)} | \
            {(x, height-1) for x in range(width)} | \
            {(0, y) for y in range(height)} | \
            {(width-1, y) for y in range(height)}

    # generate a border on the left maze side by sampling the upper half and
    # mirroring that to the lower half
    #
    # the border is located on the right edge of the left maze side
    pos = width // 2 - 1

    # sample the upper y-coordinates
    ys = rng.sample(range(height // 2), k=(height - ngaps_center) // 2)

    # mirror the y-coordinates to the lower half
    ys += [height - 1 - y for y in ys]

    # add the border walls to the maze edges
    border = set((pos, y) for y in ys)
    walls |= border

    # prepare left maze side as partition
    pmin = (0, 0)
    pmax = (pos, height - 1)
    partition = Partition(pmin, pmax, vertical=False, rng=rng)

    # add inner walls to the left maze side
    walls = add_walls(
        partition,
        walls,
        ngaps_center // 2,
        rng=rng,
    )

    # make space for the pacmen
    walls -= bots_pos

    return walls


def generate_maze(trapped_food=10, total_food=30, width=32, height=16, rng=None):
    if width % 2 != 0:
        raise ValueError(f"Width must be even ({width} given)")

    if width < 4:
        raise ValueError(f"Width must be at least 4, but {width} was given")

    if height < 4:
        raise ValueError(f"Height must be at least 4, but {height} was given")

    rng = default_rng(rng)

    # generate a full maze, but only the left half is filled with random walls
    # this allows us to cut the execution time in two, because the following
    # graph operations are quite expensive
    pacmen_pos = set([(1, height - 3), (1, height - 2)])
    walls = generate_half_maze(width, height, height//2, pacmen_pos, rng=rng)

    ### TODO: hide the chamber_finding in another function, create the graph with
    # a wall on the right border + 1, so that find chambers works reliably and
    # we can get rid of the  {.... if tile[0] < border} in the following
    # also, improve find_trapped_tiles so that it does not use x and width, but just
    # requires two sets of nodes representing the left and the right of the border
    # and then the main chambers is that one that has a non-empty intersection
    # with both.

    # transform to graph to find dead ends and chambers for food distribution
    # IMPORTANT: we have to include one column of the right border in the graph
    # generation, or our algorithm to find chambers would get confused
    # Note: this only works because in the right side of the maze we have no walls
    # except for the surrounding ones.
    graph = walls_to_graph(walls, shape=(width//2+1, height))

    # the algorithm should actually guarantee this, but just to make sure, let's
    # fail if the graph is not fully connected
    if not nx.is_connected(graph):
        raise ValueError("Generated maze is not fully connected, try a different random seed")

    # this gives us a set of tiles that are "trapped" within chambers, i.e. tunnels
    # with a dead-end or a section of tiles fully enclosed by walls except for a single
    # tile entrance
    chamber_tiles, _ = find_trapped_tiles(graph, width, include_chambers=False)

    # we want to distribute the food only on the left half of the maze
    # make sure that the tiles available for food distribution do not include
    # those right on the border of the homezone
    # also, no food on the initial positions of the pacmen
    # IMPORTANT: the relevant chamber tiles are only those in the left side of
    # the maze. By detecting chambers on only half of the maze, we may still have
    # spurious chambers on the right side
    border = width//2 - 1
    chamber_tiles = {tile for tile in chamber_tiles if tile[0] < border} - pacmen_pos
    all_tiles = {(x, y) for x in range(border) for y in range(height)}
    free_tiles = all_tiles - walls - pacmen_pos
    left_food = distribute_food(free_tiles, chamber_tiles, trapped_food, total_food, rng=rng)

    # get the full maze with all walls and food by mirroring the left half
    food = mirror(left_food, width, height)
    walls = mirror(walls, width, height)
    layout = { "walls" : tuple(sorted(walls)),
               "food"  : sorted(food),
               "bots"  : [ (1, height - 3), (width - 2, 2),
                           (1, height - 2), (width - 2, 1) ],
               "shape" : (width, height) }

    return layout
