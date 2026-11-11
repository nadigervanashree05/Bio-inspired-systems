import random
import matplotlib.pyplot as plt

# Objective function
def fitness(x, y):
    return x**2 + y**2


# PSO parameters
NUM_PARTICLES = 30
ITERATIONS = 100
W = 0.7
C1 = 1.5
C2 = 1.5

random.seed(42)

# Initialize particle positions
positions = [
    [random.uniform(-10, 10), random.uniform(-10, 10)]
    for _ in range(NUM_PARTICLES)
]

# Initialize particle velocities
velocities = [
    [random.uniform(-1, 1), random.uniform(-1, 1)]
    for _ in range(NUM_PARTICLES)
]

# Initialize personal best positions
pbest = [p[:] for p in positions]

pbest_fitness = [
    fitness(p[0], p[1]) for p in pbest
]

# Initialize global best
best_index = pbest_fitness.index(min(pbest_fitness))

gbest = pbest[best_index][:]
gbest_fitness = pbest_fitness[best_index]

# Store convergence history
history = []

# PSO algorithm
for iteration in range(ITERATIONS):

    for i in range(NUM_PARTICLES):

        for d in range(2):

            r1 = random.random()
            r2 = random.random()

            # Update velocity
            velocities[i][d] = (
                W * velocities[i][d]
                + C1 * r1 * (pbest[i][d] - positions[i][d])
                + C2 * r2 * (gbest[d] - positions[i][d])
            )

            # Limit velocity
            velocities[i][d] = max(
                -2, min(2, velocities[i][d])
            )

            # Update position
            positions[i][d] += velocities[i][d]

            # Keep positions within bounds
            positions[i][d] = max(
                -10, min(10, positions[i][d])
            )

        # Calculate current fitness
        current_fitness = fitness(
            positions[i][0],
            positions[i][1]
        )

        # Update personal best
        if current_fitness < pbest_fitness[i]:

            pbest[i] = positions[i][:]

            pbest_fitness[i] = current_fitness

    # Find the best personal best
    best_index = pbest_fitness.index(
        min(pbest_fitness)
    )

    # Update global best
    if pbest_fitness[best_index] < gbest_fitness:

        gbest = pbest[best_index][:]
        gbest_fitness = pbest_fitness[best_index]

    # Store best fitness
    history.append(gbest_fitness)


# Display final results
print("\n========== PSO RESULTS ==========")

print(f"Optimal x: {gbest[0]:.15f}")
print(f"Optimal y: {gbest[1]:.15f}")
print(f"Minimum fitness: {gbest_fitness:.15e}")

print("\n======= FITNESS PER ITERATION =======")
print(f"{'Iteration':<12}{'Best Fitness':>25}")

for i, value in enumerate(history, start=1):
    print(f"{i:<12}{value:>25.15e}")


# Plot convergence graph
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, ITERATIONS + 1),
    history,
    color="blue",
    linewidth=2,
    marker="o",
    markersize=3,
    label="Global Best Fitness"
)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("PSO Convergence Graph")

# Display small values in scientific notation
plt.ticklabel_format(
    axis="y",
    style="sci",
    scilimits=(0, 0)
)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show() 