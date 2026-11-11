import numpy as np
import matplotlib.pyplot as plt

# Generate 20 random cities
np.random.seed(42)
N = 20
cities = np.random.rand(N, 2) * 100

# Calculate distance and visibility matrices
diff = cities[:, None, :] - cities[None, :, :]
distance = np.sqrt(np.sum(diff**2, axis=2))

visibility = np.zeros_like(distance)
np.divide(1, distance, out=visibility, where=distance > 0)

# Ant System parameters
n_ants = N
iterations = 300
alpha = 1.0
beta = 5.0
p = 0.5
Q = 100.0

pheromone = np.ones((N, N))
np.fill_diagonal(pheromone, 0)

best_tour = None
best_length = float("inf")

# Calculate tour length
def tour_length(tour):
    return sum(
        distance[tour[i], tour[(i + 1) % N]]
        for i in range(N)
    )

# Run Ant System
for iteration in range(iterations):
    tours = []
    lengths = []

    for start in range(n_ants):
        tour = [start]
        visited = {start}

        while len(tour) < N:
            current = tour[-1]

            candidates = np.array(
                [c for c in range(N) if c not in visited]
            )

            weights = (
                pheromone[current, candidates] ** alpha
                * visibility[current, candidates] ** beta
            )

            probabilities = weights / weights.sum()

            next_city = np.random.choice(
                candidates, p=probabilities
            )

            tour.append(int(next_city))
            visited.add(int(next_city))

        length = tour_length(tour)
        tours.append(tour)
        lengths.append(length)

        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    # Update pheromones
    deposits = np.zeros((N, N))

    for tour, length in zip(tours, lengths):
        amount = Q / length

        for i in range(N):
            a = tour[i]
            b = tour[(i + 1) % N]

            deposits[a, b] += amount
            deposits[b, a] += amount

    pheromone = p * pheromone + deposits
    np.fill_diagonal(pheromone, 0)

# Print output
route = best_tour + [best_tour[0]]

print("Best Tour:", " → ".join(map(str, route)))
print(f"Best Tour Length: {best_length:.1f} units")

# Plot best route
path = cities[route]

plt.figure(figsize=(9, 7))

plt.plot(
    path[:, 0], path[:, 1],
    "b-o", linewidth=1.5, markersize=5
)

plt.scatter(
    cities[:, 0], cities[:, 1],
    color="red", zorder=3
)

# Display city numbers
for i, (x, y) in enumerate(cities):
    plt.annotate(
        str(i), (x, y),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title(f"Ant System - Best Tour (Length = {best_length:.1f})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()