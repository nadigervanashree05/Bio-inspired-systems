
import numpy as np
import random

# 1. Optimization Function (Sphere Function)
def optimization_function(solution):
    return sum(x**2 for x in solution)


# 2. Initialize Parameters
POPULATION_SIZE = 50
GENES = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 30


# 3. Initialize Population
def initialize_population(pop_size, genes):
    return np.random.uniform(-10, 10, (pop_size, genes))


# 4. Evaluate Fitness
def evaluate_fitness(population):
    fitness = [
        optimization_function(solution)
        for solution in population
    ]
    return np.array(fitness)


# 5. Selection (Tournament Selection)
def select_parents(population, fitness, num_parents):
    parents = []

    for _ in range(num_parents):
        tournament = random.sample(
            range(len(population)), 3
        )

        best = min(tournament, key=lambda idx: fitness[idx])
        parents.append(population[best])

    return np.array(parents)


# 6. Crossover (Single-Point Crossover)
def crossover(parents, crossover_rate):
    offspring = []

    for i in range(0, len(parents) - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        if random.random() < crossover_rate:
            point = random.randint(1, len(parent1) - 1)

            child1 = np.concatenate(
                (parent1[:point], parent2[point:])
            )

            child2 = np.concatenate(
                (parent2[:point], parent1[point:])
            )
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        offspring.extend([child1, child2])

    return np.array(offspring)


# 7. Mutation
def mutate(offspring, mutation_rate):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if random.random() < mutation_rate:
                offspring[i][j] += np.random.uniform(-1, 1)

    return offspring


# 8. Gene Expression
def gene_expression(population):
    return population


# 9. Main Function
def gene_expression_algorithm():

    # Initialize population
    population = initialize_population(
        POPULATION_SIZE, GENES
    )

    best_solution = None
    best_fitness = float('inf')

    for generation in range(GENERATIONS):

        # Evaluate fitness
        fitness = evaluate_fitness(population)

        # Track best solution
        min_fitness_idx = np.argmin(fitness)

        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_solution = population[min_fitness_idx].copy()

        # Selection
        parents = select_parents(
            population, fitness, POPULATION_SIZE // 2
        )

        # Crossover
        offspring = crossover(parents, CROSSOVER_RATE)

        # Mutation
        offspring = mutate(offspring, MUTATION_RATE)

        # Gene Expression
        population = gene_expression(offspring)

        # Print progress
        print(
            f"Generation {generation + 1}: "
            f"Best Fitness = {best_fitness}"
        )

    # Output best solution
    print("\nBest Solution Found:")
    print("Position:", best_solution)
    print("Fitness:", best_fitness)


# 10. Run the Algorithm
if __name__ == "__main__":
    gene_expression_algorithm()
