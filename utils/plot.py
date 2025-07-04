import matplotlib.pyplot as plt

def plot_routes(instance, routes, title="CVRP Solution"):
    coords = instance["node_coord"]         # shape: (n, 2), with depot at index 0
    depot = coords[0]

    plt.figure(figsize=(8, 6))
    plt.title(title)

    # Plot depot
    plt.scatter(*depot, c='red', marker='s', s=100, label="Depot")

    # Plot all customers
    plt.scatter(coords[1:, 0], coords[1:, 1], c='black', s=30, label="Customers")

    # Plot routes
    for i, route in enumerate(routes):
        route_coords = [depot] + [coords[c] for c in route] + [depot]
        x, y = zip(*route_coords)
        plt.plot(x, y, marker='o', label=f"Route {i+1}")

    # Optional: label customer IDs
    for i, (x, y) in enumerate(coords):
        label = str(i) if i > 0 else "D"
        plt.text(x + 1, y + 1, label, fontsize=8)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
