import numpy as np

# Optimization Function
def optimization_function(position):
    return sum(x**2 for x in position)


# Parameters
GRID_SIZE = (10, 10)
DIMENSIONS = 2
ITERATIONS = 30
NEIGHBORHOOD_RADIUS = 1

np.random.seed(10)


# Initialize Population
def initialize_population(grid_size, dimensions):
    return np.random.uniform(
        -10, 10,
        size=(grid_size[0], grid_size[1], dimensions)
    )


# Evaluate Fitness
def evaluate_fitness(population):
    return np.sum(population**2, axis=2)


# Get Neighbourhood
def get_neighborhood(grid, x, y, radius):
    neighbors = []

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):

            if i == 0 and j == 0:
                continue

            ni, nj = x + i, y + j

            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                neighbors.append((ni, nj))

    return neighbors


# Update States
def update_states(population, fitness):
    new_population = np.copy(population)

    for i in range(population.shape[0]):
        for j in range(population.shape[1]):

            neighbors = get_neighborhood(
                population, i, j, NEIGHBORHOOD_RADIUS
            )

            best_neighbor = population[i, j]
            best_fitness = fitness[i, j]

            for ni, nj in neighbors:
                if fitness[ni, nj] < best_fitness:
                    best_fitness = fitness[ni, nj]
                    best_neighbor = population[ni, nj]

            new_population[i, j] = (
                population[i, j] + best_neighbor
            ) / 2

    return new_population


# Main Algorithm
def parallel_cellular_algorithm():

    population = initialize_population(GRID_SIZE, DIMENSIONS)

    best_solution = None
    best_fitness = float('inf')

    for iteration in range(ITERATIONS):

        fitness = evaluate_fitness(population)

        min_fitness = np.min(fitness)

        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[
                np.unravel_index(np.argmin(fitness), fitness.shape)
            ].copy()

        population = update_states(population, fitness)

        print(
            f"Iteration {iteration + 1}: "
            f"Best Fitness = {best_fitness:.6f}"
        )

    print("\nBest Solution Found:")
    print("Position:", best_solution)
    print("Fitness:", best_fitness)


# Run Algorithm
if __name__ == "__main__":
    parallel_cellular_algorithm()
