import random
import matplotlib.pyplot as plt
import numpy as np


class ChordNode:
    def __init__(self, node_id, m=7):
        """
        Initialize a Chord node.

        param node_id: The unique identifier for this node.
        param m: The number of bits in the identifier space.
        """
        self.node_id = node_id
        self.m = m
        self.max_id = 2**m
        self.finger = [None]*(m)
        self.successor = None
        self.predecessor = None

    def in_range(self, x, start, end, inclusive_end=False):
        """
        Checks if x is in the range (start, end) on a circle of size max_id.
        """
        if start < end:
            return start < x < end if not inclusive_end else start < x <= end
        else:
            return x > start or x < end if not inclusive_end else x > start or x <= end

    def find_successor(self, identifier, count_hops=False):
        """
        Find the successor of 'identifier'.
        If count_hops is True, also count how many hops we take.
        """
        if count_hops:
            predecessor, hops = self.find_predecessor(
                identifier, count_hops=True)
            return predecessor.successor, hops + 1
        else:
            predecessor = self.find_predecessor(identifier)
            return predecessor.successor

    def find_predecessor(self, identifier, count_hops=False):
        """
        Find the predecessor of 'identifier'.
        If count_hops is True, returns (predecessor_node, hops).
        """
        node = self
        hops = 0
        while not node.in_range(identifier, node.node_id, node.successor.node_id, inclusive_end=True):
            node = node.closest_preceding_finger(identifier)
            if count_hops:
                hops += 1
        if count_hops:
            return node, hops
        else:
            return node

    def closest_preceding_finger(self, identifier):
        """
        Return the closest finger node that precedes 'identifier'.
        """
        for i in reversed(range(self.m)):
            if self.finger[i] and self.in_range(self.finger[i].node_id, self.node_id, identifier):
                return self.finger[i]
        return self

    def join(self, existing_node):
        """
        Join the ring given a known node 'existing_node'.
        """
        if existing_node is not None:
            self.init_finger_table(existing_node)
            self.update_others()
        else:
            for i in range(self.m):
                self.finger[i] = self
            self.predecessor = self
            self.successor = self

    def init_finger_table(self, existing_node):
        self.finger[0] = existing_node.find_successor(
            (self.node_id + 2**0) % self.max_id)
        if isinstance(self.finger[0], tuple):
            # if find_successor returns (node, hops), extract node part
            self.finger[0] = self.finger[0][0]
        self.successor = self.finger[0]
        self.predecessor = self.successor.predecessor
        self.successor.predecessor = self
        for i in range(self.m-1):
            start = (self.node_id + 2**(i+1)) % self.max_id
            if self.in_range(start, self.node_id, self.finger[i].node_id, inclusive_end=True):
                self.finger[i+1] = self.finger[i]
            else:
                suc = existing_node.find_successor(start)
                if isinstance(suc, tuple):
                    suc = suc[0]
                self.finger[i+1] = suc

    def update_others(self):
        for i in range(self.m):
            start = (self.node_id - 2**i) % self.max_id
            p = self.find_predecessor_of_id(start)
            p.update_finger_table(self, i)

    def find_predecessor_of_id(self, identifier):
        return self.find_predecessor(identifier)

    def update_finger_table(self, s, i):
        start = (self.node_id + 2**i) % self.max_id
        if (self.finger[i] == self) or self.in_range(s.node_id, self.node_id, self.finger[i].node_id, inclusive_end=False):
            self.finger[i] = s
            if i == 0:
                self.successor = s
            self.predecessor.update_finger_table(s, i)

    def lookup(self, key):
        """
        Lookup a key and return (responsible_node, hops).
        """
        node, hops = self.find_successor(key, count_hops=True)
        return node, hops


def compute_spectral_gap(nodes):
    """
    Compute the spectral gap of the graph represented by the Chord nodes.
    The spectral gap is the difference between the first two smallest eigenvalues of the Laplacian matrix.
    """
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n))

    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if other_node in node.finger or other_node == node.successor:
                adjacency_matrix[i, j] = 1

    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix

    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)

    spectral_gap = eigenvalues[1] - eigenvalues[0]
    return spectral_gap


def simulate_2():
    """
    Simulate Chord networks with varying levels of expander strength and compare lookup times.
    """
    m = 7  # num bits in identifier space
    num_nodes = 50  # num nodes in each ring
    num_lookups = 100
    expander_levels = 5

    all_hops = []
    labels = []

    additional_neighbors_by_level = [0, 2, 5, 10, 20]

    for level, extra_neighbors in enumerate(additional_neighbors_by_level):
        node_ids = sorted(random.sample(range(2**m), num_nodes))
        nodes = [ChordNode(node_ids[0], m=m)]
        nodes[0].join(None)
        for nid in node_ids[1:]:
            new_node = ChordNode(nid, m=m)
            new_node.join(nodes[0])
            nodes.append(new_node)

        for node in nodes:
            additional_neighbors = random.sample(
                nodes, k=min(extra_neighbors, len(nodes) - 1))
            for neighbor in additional_neighbors:
                if neighbor != node and neighbor not in node.finger:
                    node.finger.append(neighbor)

        spectral_gap = compute_spectral_gap(nodes)

        hops = []
        for _ in range(num_lookups):
            key = random.randint(0, (2**m)-1)
            start_node = random.choice(nodes)
            _, hop_count = start_node.lookup(key)
            hops.append(hop_count)

        all_hops.append(hops)
        labels.append(f"Level {level + 1} (Gap: {spectral_gap:.4f})")

    plt.figure(figsize=(14, 7))

    for level, hops in enumerate(all_hops):
        freq, bins = np.histogram(hops, bins=range(1, max(hops) + 2))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.scatter(bin_centers, freq, alpha=0.6)
        plt.plot(bin_centers, freq, linestyle='-', marker='o',
                 label=f"{labels[level]}")

    plt.title('Chord Lookup Times For Varying Expansions Levels')
    plt.xlabel('Lookup Time (Hops)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    m = 7
    node0 = ChordNode(node_id=10, m=m)
    node0.join(existing_node=None)

    node_ids = [20, 45, 87, 3, 60, 90, 110]
    nodes = [node0]
    for nid in node_ids:
        new_node = ChordNode(nid, m=m)
        new_node.join(node0)
        nodes.append(new_node)

    num_lookups = 500
    hop_counts = []
    for _ in range(num_lookups):
        key = random.randint(0, (2**m)-1)
        _, hops = node0.lookup(key)
        hop_counts.append(hops)

    plt.figure(figsize=(10, 6))
    bins = range(min(hop_counts), max(hop_counts)+2)
    freq, _, _ = plt.hist(hop_counts, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Lookup Times (Hops) in Chord DHT')
    plt.xlabel('Lookup Time (Hops)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    simulate_2()
