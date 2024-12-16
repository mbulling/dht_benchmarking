import random
import math
import matplotlib.pyplot as plt
import numpy as np


class CANNode:
    def __init__(self, xmin, xmax, ymin, ymax):
        """
        A CAN node responsible for a zone defined by the rectangle [xmin,xmax] x [ymin,ymax].
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.neighbors = []

    def center(self):
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def is_responsible(self, x, y):
        """
        Check if this node's zone is responsible for the point (x, y).
        """
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def distance_to(self, x, y):
        """
        Compute Euclidean distance from the center of this node's zone to (x, y).
        """
        cx, cy = self.center()
        dx = x - cx
        dy = y - cy
        return math.sqrt(dx*dx + dy*dy)

    def split_zone(self):
        """
        Split this node's zone into two halves along the longer dimension.
        Return (new_node, dimension) where dimension is 'x' or 'y' indicating the split direction.
        """
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        if width >= height:
            mid = (self.xmin + self.xmax) / 2.0
            new_node = CANNode(mid, self.xmax, self.ymin, self.ymax)
            self.xmax = mid
            return new_node, 'x'
        else:
            mid = (self.ymin + self.ymax) / 2.0
            new_node = CANNode(self.xmin, self.xmax, mid, self.ymax)
            self.ymax = mid
            return new_node, 'y'


def add_neighbor_relation(node1, node2):
    """
    Add node2 as a neighbor of node1 if they share a boundary.
    """
    if are_neighbors(node1, node2) and node2 not in node1.neighbors:
        node1.neighbors.append(node2)
    if are_neighbors(node2, node1) and node1 not in node2.neighbors:
        node2.neighbors.append(node1)


def are_neighbors(n1, n2):
    """
    Two nodes are neighbors if their zones share a boundary region.
    """
    overlap_x = (n1.xmin <= n2.xmax and n2.xmin <= n1.xmax)
    overlap_y = (n1.ymin <= n2.ymax and n2.ymin <= n1.ymax)
    if overlap_x and overlap_y:
        touch_x = n1.xmax == n2.xmin or n2.xmax == n1.xmin
        touch_y = n1.ymax == n2.ymin or n2.ymax == n1.ymin
        return (touch_x or touch_y)
    return False


def update_neighbors(nodes):
    """
    Update all neighbor relationships after a split.
    """
    for n in nodes:
        n.neighbors = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            add_neighbor_relation(nodes[i], nodes[j])


def find_node_for_point(nodes, x, y):
    """
    Find the node responsible for (x, y).
    """
    for node in nodes:
        if node.is_responsible(x, y):
            return node
    return None


def join_node(nodes):
    """
    A new node joins the CAN
    - pick random point (x, y) in [0,1]^2
    - find node responsible for (x, y)
    - split that node's zone and add the new node
    - update neighbors
    """
    x, y = random.random(), random.random()
    target = find_node_for_point(nodes, x, y)
    if target is None:
        return
    new_node, dim = target.split_zone()
    nodes.append(new_node)
    update_neighbors(nodes)


def lookup(nodes, start_node, x, y):
    """
    Perform a lookup from start_node to find the node responsible for (x, y)
    Returns (responsible_node, hops)
    """
    current = start_node
    hops = 0
    while not current.is_responsible(x, y):
        hops += 1
        best = current
        best_dist = current.distance_to(x, y)
        for nbr in current.neighbors:
            d = nbr.distance_to(x, y)
            if d < best_dist:
                best = nbr
                best_dist = d
        if best == current:
            break
        current = best
    return current, hops


def simulate_expander_graphs():
    """
    Simulate networks with weak and strong expander graphs and compare lookup times.
    """
    num_nodes = 50
    num_lookups = 100
    expander_levels = 5

    all_hops = []
    labels = []

    for level in range(expander_levels):
        nodes = [CANNode(0.0, 1.0, 0.0, 1.0)]
        for _ in range(num_nodes):
            join_node(nodes)

        if level > 0:
            for node in nodes:
                additional_neighbors = random.sample(
                    nodes, k=min(level, len(nodes) - 1))
                for neighbor in additional_neighbors:
                    if neighbor != node:
                        add_neighbor_relation(node, neighbor)

        hops = []
        for _ in range(num_lookups):
            x, y = random.random(), random.random()
            start_node = random.choice(nodes)
            _, hop_count = lookup(nodes, start_node, x, y)
            hops.append(hop_count)

        all_hops.append(hops)
        labels.append(f"Level {level + 1}")

    plt.figure(figsize=(14, 7))

    for level, hops in enumerate(all_hops):
        freq, bins = np.histogram(hops, bins=range(1, max(hops) + 2))
        plt.scatter(bins[:-1], freq, alpha=0.6, label=labels[level])
        trend = np.poly1d(np.polyfit(bins[:-1], freq, 1))
        plt.plot(bins[:-1], trend(bins[:-1]), linestyle='--',
                 label=f"{labels[level]} Trend Line")

    plt.title('Lookup Times Across Varying Expander Strengths')
    plt.xlabel('Lookup Time (Hops)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()


def compute_spectral_gap(nodes):
    """
    Compute the spectral gap of the graph represented by the CAN nodes.
    """
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n))

    for i, node in enumerate(nodes):
        for neighbor in node.neighbors:
            j = nodes.index(neighbor)
            adjacency_matrix[i, j] = 1

    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix

    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues = np.sort(np.real(eigenvalues))

    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    spectral_gap = eigenvalues[-1] - \
        eigenvalues[-2] if len(eigenvalues) > 1 else 0

    return spectral_gap


def simulate_2():
    """
    Simulate networks with varying levels of expander strength.
    Levels are correlated with spectral gap: Level 1 = smallest gap, Level 5 = largest gap.
    Spectral gaps are adjusted to range between 5 and 10.
    """
    num_nodes = 50
    num_lookups = 100

    all_hops = []
    labels = []

    additional_neighbors_by_level = [0, 2, 5, 10, 20]
    spectral_gaps = []

    for level, extra_neighbors in enumerate(additional_neighbors_by_level):
        nodes = [CANNode(0.0, 1.0, 0.0, 1.0)]
        for _ in range(num_nodes):
            join_node(nodes)

        for node in nodes:
            additional_neighbors = random.sample(
                nodes, k=min(extra_neighbors, len(nodes) - 1))
            for neighbor in additional_neighbors:
                if neighbor != node:
                    add_neighbor_relation(node, neighbor)

        spectral_gap = compute_spectral_gap(nodes)
        spectral_gaps.append(spectral_gap)

        hops = []
        for _ in range(num_lookups):
            x, y = random.random(), random.random()
            start_node = random.choice(nodes)
            _, hop_count = lookup(nodes, start_node, x, y)
            hops.append(hop_count)

        all_hops.append(hops)
        labels.append(f"Level {level + 1} (Gap: {spectral_gap:.2f})")

    plt.figure(figsize=(14, 7))

    d = {}
    for i in range(len(spectral_gaps)):
        d[spectral_gaps[i]] = i

    indices = {}
    spectral_gaps = sorted(spectral_gaps)
    for i in range(len(spectral_gaps)):
        indices[d[spectral_gaps[i]]] = i

    for level, hops in enumerate(all_hops):
        freq, bins = np.histogram(hops, bins=range(1, max(hops) + 2))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.scatter(bin_centers, freq, alpha=0.6)
        plt.plot(bin_centers, freq, linestyle='-', marker='o',
                 label=f"{labels[level]}")

    plt.title('CAN Lookup Times for Varying Expansion Levels')
    plt.xlabel('Lookup Time (Hops)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    nodes = [CANNode(0.0, 1.0, 0.0, 1.0)]

    for _ in range(20):
        join_node(nodes)

    num_lookups = 500
    hop_counts = []
    for _ in range(num_lookups):
        x, y = random.random(), random.random()
        start_node = random.choice(nodes)
        _, hops = lookup(nodes, start_node, x, y)
        hop_counts.append(hops)

    plt.figure(figsize=(10, 6))
    bins = range(min(hop_counts), max(hop_counts)+2)
    freq, _, _ = plt.hist(hop_counts, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Lookup Times (Hops) in CAN DHT')
    plt.xlabel('Lookup Time (Hops)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    simulate_2()
