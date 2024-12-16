import random
import matplotlib.pyplot as plt
import numpy as np

m = 8  # number of bits for node ids
base = 16

def id_to_str(identifier):
    length = (m + 3) // 4
    return f"{identifier:0{length}x}"


def prefix_length(a, b):
    length = 0
    for x, y in zip(a, b):
        if x == y:
            length += 1
        else:
            break
    return length


class PastryNode:
    def __init__(self, node_id_str):
        self.node_id_str = node_id_str
        self.m = len(node_id_str)
        self.routing_table = [{} for _ in range(self.m)]

    def add_to_routing_table(self, node):
        """
        Add 'node' to this node's routing table if it helps improve prefix coverage.
        """
        p = prefix_length(self.node_id_str, node.node_id_str)
        if p < self.m:
            next_digit = node.node_id_str[p]
            self.routing_table[p][next_digit] = node

    def route(self, key_id_str):
        """
        Route a message to 'key_id_str' using prefix-based routing.
        Return (destination_node, hops).
        """
        current = self
        hops = 0
        while current.node_id_str != key_id_str:
            hops += 1
            p = prefix_length(current.node_id_str, key_id_str)
            if p == current.m:
                break
            next_digit = key_id_str[p]
            if next_digit in current.routing_table[p]:
                current = current.routing_table[p][next_digit]
            else:
                if current.routing_table[p]:
                    candidates = list(current.routing_table[p].values())
                    current_id = int(current.node_id_str, 16)
                    key_id = int(key_id_str, 16)
                    best = min(candidates, key=lambda c: abs(
                        int(c.node_id_str, 16)-key_id))
                    current = best
                else:
                    break
            if hops > 2*self.m:
                break
        return current, hops


def pastry_join(known_node, new_node):
    """
    Simplified join: the new node contacts the known_node and integrates its routing info.
    Then we iteratively update routing tables of the known_node and any reachable nodes.
    """
    known_node.add_to_routing_table(new_node)
    new_node.add_to_routing_table(known_node)

    to_visit = []
    for row in known_node.routing_table:
        for nxt in row.values():
            to_visit.append(nxt)

    visited = set([known_node.node_id_str])
    while to_visit:
        node = to_visit.pop()
        if node.node_id_str in visited:
            continue
        visited.add(node.node_id_str)
        node.add_to_routing_table(new_node)
        new_node.add_to_routing_table(node)

        for row in node.routing_table:
            for nxt in row.values():
                if nxt.node_id_str not in visited:
                    to_visit.append(nxt)


def create_initial_network():
    node_id = random.randint(0, 2**m - 1)
    node = PastryNode(id_to_str(node_id))
    return [node]


def add_nodes(nodes, count):
    for _ in range(count):
        new_id = random.randint(0, 2**m - 1)
        new_node = PastryNode(id_to_str(new_id))
        known_node = random.choice(nodes)
        pastry_join(known_node, new_node)
        nodes.append(new_node)


def compute_spectral_gap(nodes):
    """
    Compute the spectral gap of the Pastry network based on its routing table connections.
    """
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n))

    for i, node in enumerate(nodes):
        for row in node.routing_table:
            for neighbor in row.values():
                j = nodes.index(neighbor)
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian = degree_matrix - adjacency_matrix

    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)

    spectral_gap = eigenvalues[1] - eigenvalues[0]
    return spectral_gap


def simulate_2():
    """
    Simulate Pastry networks with varying levels of connectivity and compare lookup times.
    """
    num_nodes = 50
    num_lookups = 100
    expander_levels = 5

    all_hops = []
    labels = []

    additional_neighbors_by_level = [0, 2, 5, 10, 20]

    for level, extra_neighbors in enumerate(additional_neighbors_by_level):
        nodes = create_initial_network()
        add_nodes(nodes, num_nodes - 1)
        for node in nodes:
            other_nodes = [n for n in nodes if n != node]
            additional_neighbors = random.sample(
                other_nodes, k=min(extra_neighbors, len(other_nodes))
            )
            for neighbor in additional_neighbors:
                node.add_to_routing_table(neighbor)

        spectral_gap = compute_spectral_gap(nodes)

        hops = []
        for _ in range(num_lookups):
            key_id = random.randint(0, 2**m - 1)
            key_str = id_to_str(key_id)
            start_node = random.choice(nodes)
            _, hop_count = start_node.route(key_str)
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

    plt.title("Pastry Lookup Times For Varying Expansion Levels")
    plt.xlabel("Lookup Time (Hops)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    nodes = create_initial_network()
    add_nodes(nodes, 20)  # 21 nodes total

    num_lookups = 500
    hop_counts = []
    for _ in range(num_lookups):
        start = random.choice(nodes)
        key_id = random.randint(0, 2**m - 1)
        key_str = id_to_str(key_id)
        _, hops = start.route(key_str)
        hop_counts.append(hops)

    # plot distribution of hop counts
    plt.figure(figsize=(10, 6))
    if hop_counts:
        bins = range(min(hop_counts), max(hop_counts)+2)
        plt.hist(hop_counts, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Lookup Times (Hops) in Pastry DHT (Simplified)')
    plt.xlabel('Lookup Time (Hops)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    simulate_2()
