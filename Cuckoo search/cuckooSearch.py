import numpy as np
from math import gamma, sin, pi

np.random.seed(42)

# Cuckoo Search parameters
n_nests = 25
iterations = 1000
pa = 0.25
beta = 1.5

# Spring design limits
lb = np.array([0.05, 0.25, 2.0])
ub = np.array([2.0, 1.3, 15.0])

# Objective function: Minimize spring weight
def spring_weight(x):
    w, d, L = x
    return (L + 2) * w**2 * d

# Design constraints
def constraints(x):
    w, d, L = x

    g1 = 1 - (d**3 * L) / (71785 * w**4)
    g2 = 1 - (140.45 * w) / (d**2 * L)
    g3 = 2 * (w + d) / 3 - 1
    g4 = (d * (4*d - w)) / (w**3 * (12566*d - w)) \
         + 1 / (5108*w**2) - 1

    return np.array([g1, g2, g3, g4])

# Fitness function
def fitness(x):
    penalty = np.sum(np.maximum(0, constraints(x))**2)
    return spring_weight(x) + 1e7 * penalty

# Levy flight
def levy_flight(size):
    sigma = (
        gamma(1 + beta) * sin(pi * beta / 2) /
        (gamma((1 + beta) / 2) * beta *
         2**((beta - 1) / 2))
    )**(1 / beta)

    u = np.random.randn(size) * sigma
    v = np.random.randn(size)

    return u / (np.abs(v) + 1e-12)**(1 / beta)

# Initialize nests
nests = lb + np.random.rand(n_nests, 3) * (ub - lb)
scores = np.array([fitness(x) for x in nests])

best_index = np.argmin(scores)
best_solution = nests[best_index].copy()
best_score = scores[best_index]

# Cuckoo Search algorithm
for iteration in range(iterations):

    for i in range(n_nests):

        step = levy_flight(3)

        new_solution = nests[i] + \
            0.01 * step * (nests[i] - best_solution) * \
            np.random.randn(3)

        new_solution = np.clip(new_solution, lb, ub)
        new_score = fitness(new_solution)

        j = np.random.randint(n_nests)

        if new_score < scores[j]:
            nests[j] = new_solution
            scores[j] = new_score

    # Replace some nests
    for i in range(n_nests):

        if np.random.rand() < pa:

            a, b = np.random.choice(n_nests, 2, replace=False)

            new_solution = nests[i] + \
                np.random.rand(3) * (nests[a] - nests[b])

            new_solution = np.clip(new_solution, lb, ub)
            new_score = fitness(new_solution)

            if new_score < scores[i]:
                nests[i] = new_solution
                scores[i] = new_score

    # Update best solution
    current_best = np.argmin(scores)

    if scores[current_best] < best_score:
        best_score = scores[current_best]
        best_solution = nests[current_best].copy()

# Display final output
w, d, L = best_solution

print("Best Spring Design:")
print(f"Wire Diameter: {w:.6f}")
print(f"Coil Diameter: {d:.6f}")
print(f"Number of Coils: {L:.6f}")
print(f"Minimum Spring Weight: {spring_weight(best_solution):.6f}")
print("Constraints Satisfied:",
      np.all(constraints(best_solution) <= 1e-6))