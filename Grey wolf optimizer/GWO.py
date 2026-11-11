import numpy as np

# Set random seed
np.random.seed(42)

# Task lengths (million instructions)
tasks = np.array([1200, 2500, 1800, 3000, 1500,
                  2200, 2800, 1700, 3200, 2000])

# Virtual machine speeds (MIPS)
vm_speeds = np.array([500, 1000, 1500])

# Energy consumption rate of each VM
vm_energy = np.array([100, 150, 200])

# GWO parameters
n_wolves = 20
iterations = 100
n_tasks = len(tasks)
n_vms = len(vm_speeds)

# Fitness function
def fitness(position):

    # Convert wolf position into VM assignments
    assignment = np.clip(
        np.rint(position).astype(int), 0, n_vms - 1
    )

    vm_times = np.zeros(n_vms)
    total_energy = 0

    for i in range(n_tasks):
        vm = assignment[i]

        execution_time = tasks[i] / vm_speeds[vm]

        vm_times[vm] += execution_time

        total_energy += execution_time * vm_energy[vm]

    makespan = np.max(vm_times)

    # Combined objective function
    return makespan + 0.001 * total_energy


# Initialize grey wolves
wolves = np.random.uniform(
    0, n_vms - 1, (n_wolves, n_tasks)
)

# Store best solution
best_position = None
best_fitness = float("inf")

# Mean Grey Wolf Optimization
for iteration in range(iterations):

    # Calculate fitness of all wolves
    scores = np.array([fitness(wolf) for wolf in wolves])

    # Find Alpha, Beta, and Delta
    sorted_indices = np.argsort(scores)

    alpha = wolves[sorted_indices[0]].copy()
    beta = wolves[sorted_indices[1]].copy()
    delta = wolves[sorted_indices[2]].copy()

    # Update global best
    if scores[sorted_indices[0]] < best_fitness:
        best_fitness = scores[sorted_indices[0]]
        best_position = alpha.copy()

    # Parameter decreases from 2 to 0
    a = 2 - iteration * (2 / iterations)

    # Mean position of wolves
    mean_position = np.mean(wolves, axis=0)

    for i in range(n_wolves):

        new_positions = []

        for leader in [alpha, beta, delta]:

            r1 = np.random.rand(n_tasks)
            r2 = np.random.rand(n_tasks)

            A = 2 * a * r1 - a
            C = 2 * r2

            # Modified mean-based distance
            D = np.abs(C * leader - mean_position)

            X = leader - A * D

            new_positions.append(X)

        # Update wolf position
        wolves[i] = np.mean(new_positions, axis=0)

        # Keep positions within VM limits
        wolves[i] = np.clip(wolves[i], 0, n_vms - 1)

# Convert best solution into task assignments
assignment = np.clip(
    np.rint(best_position).astype(int), 0, n_vms - 1
)

# Calculate final results
vm_times = np.zeros(n_vms)
total_energy = 0

for i in range(n_tasks):

    vm = assignment[i]

    execution_time = tasks[i] / vm_speeds[vm]

    vm_times[vm] += execution_time

    total_energy += execution_time * vm_energy[vm]

makespan = np.max(vm_times)

# Display output
print("Best Task Scheduling:")

for i in range(n_tasks):
    print(f"Task {i + 1} -> VM {assignment[i] + 1}")

print(f"\nMinimum Makespan: {makespan:.2f} seconds")
print(f"Total Energy Consumption: {total_energy:.2f} units")