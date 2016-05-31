import numpy as np

# Graph class

class UndirectedGraph(dict):
    """Undirected graph class : hash map of Node objects"""
    def __init__(self, *args, **kwargs):
        self.n_edges = 0
        super(UndirectedGraph, self).__init__(*args, **kwargs)

    @property
    def n_nodes(self):
        return len(self)

    @n_nodes.setter
    def n_nodes(self, value):
        try:
            assert self.n_nodes == 0
        except AssertionError:
            raise Exception("nbr of nodes can only be set for empty graphs")
        self.add_nodes(range(value))

    def add_node(self, x):
        self.setdefault(x, set())

    def add_nodes(self, xs):
        for x in xs:
            self.add_node(x)

    def has_edge(self, x, y):
        return y in self.friend(x)

    def add_edge(self, x, y):
        self.add_node(x)
        self.add_node(y)
        if y not in self[x]:
            assert x not in self[y]
            self.n_edges += 1
            self[x].add(y)
            self[y].add(x)
            return True
        return False

    def add_edges(self, edges):
        return [edge for edge in edges if self.add_edge(*edge)]

    def friend(self, node):
        return self[node]

    def fof(self, node):
        return set(fof for f in self.friend(node) for fof in self.friend(f))

    def fofnotfriend(self, node):
        friends = self.friend(node)
        return [fof for f in friends for fof in self.friend(f) if fof not in
                   friends]

