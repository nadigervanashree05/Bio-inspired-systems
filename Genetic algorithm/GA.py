import random
import matplotlib.pyplot as plt

data = [1, 2, 3, 4, 10, 11, 12, 13]

POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.1
random.seed(1)

def fitness(centers):
    error = 0
    for x in data:
        error += min((x - c) ** 2 for c in centers)
    return error

population = [
    [random.uniform(min(data), max(data)),
     random.uniform(min(data), max(data))]
    for _ in range(POPULATION_SIZE)
]

best = min(population, key=fitness)[:]

for generation in range(GENERATIONS):

    population.sort(key=fitness)

    if fitness(population[0]) < fitness(best):
        best = population[0][:]

    selected = population[:POPULATION_SIZE // 2]

    new_population = [best[:]]  # Elitism

    while len(new_population) < POPULATION_SIZE:
        p1 = random.choice(selected)
        p2 = random.choice(selected)

        child = [p1[0], p2[1]]

        for i in range(2):
            if random.random() < MUTATION_RATE:
                child[i] = random.uniform(min(data), max(data))

        new_population.append(child)

    population = new_population

centers = sorted(best)

cluster1 = []
cluster2 = []

for x in data:
    if abs(x - centers[0]) <= abs(x - centers[1]):
        cluster1.append(x)
    else:
        cluster2.append(x)

if cluster1:
    centers[0] = sum(cluster1) / len(cluster1)

if cluster2:
    centers[1] = sum(cluster2) / len(cluster2)

print("Optimal Cluster Centers:", centers)
print("Cluster 1:", cluster1)
print("Cluster 2:", cluster2)
print("Final SSE:", fitness(centers))

plt.scatter(cluster1, [1] * len(cluster1),
            color="blue", s=80, label="Cluster 1")

plt.scatter(cluster2, [2] * len(cluster2),
            color="red", s=80, label="Cluster 2")

plt.scatter(centers[0], 1, color="black",
            marker="X", s=200, label="Cluster Centers")

plt.scatter(centers[1], 2, color="black",
            marker="X", s=200)

plt.xlabel("Data Points")
plt.ylabel("Clusters")
plt.yticks([1, 2], ["Cluster 1", "Cluster 2"])
plt.title("Genetic K-Means Clustering")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()