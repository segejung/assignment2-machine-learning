import numpy as np 
import mlrose_hiive
import matplotlib.pyplot as plt
import time
from random import randint
import warnings

np.random.seed(97)

fitness_simulated_annealing = []
fitness_random_hill_climb = []
fitness_genetic_algorithm = []
fitness_mimic = []

time_simulated_annealing = []
time_random_hill_climb = []
time_genetic_algorithm = []
time_mimic = []

## Plot effect of increasing problem size

range_values = range(10,175,15)

for value in range_values:
	fitness = mlrose_hiive.FlipFlop()
	problem = mlrose_hiive.DiscreteOpt(length = value, fitness_fn = fitness, maximize = True, max_val = 2)
	problem.set_mimic_fast_mode(True)
	init_state = np.random.randint(2, size = value)
	start = time.time()
	_, best_fitness_sa, _ = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
	end = time.time()
	sa_time = end - start
	print("SA:", sa_time, value)

	start = time.time()
	_, best_fitness_rhc, _ = mlrose_hiive.random_hill_climb(problem, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
	end = time.time()
	rhc_time = end - start
	print("RHC:", rhc_time, value)

	start = time.time()
	_, best_fitness_ga, _ = mlrose_hiive.genetic_alg(problem, max_attempts = 10, curve = True)
	end = time.time()
	ga_time = end - start
	print("GA:", ga_time, value)

	start = time.time()
	_, best_fitness_mimic, _ = mlrose_hiive.mimic(problem, pop_size = 300, max_attempts = 10, curve = True)
	end = time.time()
	mimic_time = end - start
	print("MIMIC:", mimic_time, value)

	fitness_simulated_annealing.append(best_fitness_sa)
	fitness_random_hill_climb.append(best_fitness_rhc)
	fitness_genetic_algorithm.append(best_fitness_ga)
	fitness_mimic.append(best_fitness_mimic)

	time_simulated_annealing.append(sa_time)
	time_random_hill_climb.append(rhc_time)
	time_genetic_algorithm.append(ga_time)
	time_mimic.append(mimic_time)

fitness_simulated_annealing = np.array(fitness_simulated_annealing)
fitness_random_hill_climb = np.array(fitness_random_hill_climb)
fitness_genetic_algorithm = np.array(fitness_genetic_algorithm)
fitness_mimic = np.array(fitness_mimic)

time_simulated_annealing = np.array(time_simulated_annealing)
time_random_hill_climb = np.array(time_random_hill_climb)
time_genetic_algorithm = np.array(time_genetic_algorithm)
time_mimic = np.array(time_mimic)

plt.figure()
plt.plot(range_values, fitness_simulated_annealing, label = 'SA')
plt.plot(range_values, fitness_random_hill_climb, label = 'RHC')
plt.plot(range_values, fitness_genetic_algorithm, label = 'GA')
plt.plot(range_values, fitness_mimic, label = 'MIMIC')
plt.title('Fitness vs Problem Size (FF)')
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('FF fitness.png')

plt.figure()
plt.plot(range_values, time_simulated_annealing, label = 'SA')
plt.plot(range_values, time_random_hill_climb, label = 'RHC')
plt.plot(range_values, time_genetic_algorithm, label = 'GA')
plt.plot(range_values, time_mimic, label = 'MIMIC')
plt.title('Computation Time vs Problem Size (FF)')
plt.legend()
plt.xlabel('Problem Size')
plt.ylabel('Computation Time (s)')
plt.savefig('FF computation.png')

## Plot change with respect to iterations

problem_length = 160
fitness = mlrose_hiive.FlipFlop()
problem = mlrose_hiive.DiscreteOpt(length = problem_length, fitness_fn = fitness, maximize = True, max_val = 2)
problem.set_mimic_fast_mode(True)
init_state = np.random.randint(2, size = problem_length)
_, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
print("Done with SA iterations!")
_, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(problem, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
print("Done with RHC iterations!")
_, _, fitness_curve_ga = mlrose_hiive.genetic_alg(problem, max_attempts = 10, curve = True)
print("Done with GA iterations!")
_, _, fitness_curve_mimic = mlrose_hiive.mimic(problem, pop_size = 300, max_attempts = 10, curve = True)
print("Done with MIMIC iterations!")

plt.figure()
plt.plot(fitness_curve_sa[:,0], label = 'SA')
plt.plot(fitness_curve_rhc[:,0], label = 'RHC')
plt.plot(fitness_curve_ga[:,0], label = 'GA')
plt.plot(fitness_curve_mimic[:,0], label = 'MIMIC')
plt.title('Fitness over iteration (FF)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.savefig('FF iterations.png')

## Plot variation in performance with changing hyper-parameters

problem_length = 40
fitness = mlrose_hiive.FlipFlop()
problem = mlrose_hiive.DiscreteOpt(length = problem_length, fitness_fn = fitness, maximize = True, max_val = 2)
problem.set_mimic_fast_mode(True)
init_state = np.random.randint(2, size = problem_length)

_, _, fitness_curve_sa_1 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
_, _, fitness_curve_sa_2 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.GeomDecay(), max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
_, _, fitness_curve_sa_3 = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ArithDecay(), max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
print("Completed SA hyper-parameter testing!")

plt.figure()
plt.plot(fitness_curve_sa_1[:,0], label = 'decay = Exponential')
plt.plot(fitness_curve_sa_2[:,0], label = 'decay = Geometric')
plt.plot(fitness_curve_sa_3[:,0], label = 'decay = Arithmetic')
plt.title('SA (FF)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.savefig('FF SA.png')

_, _, fitness_curve_rhc_1 = mlrose_hiive.random_hill_climb(problem, restarts = 0, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
_, _, fitness_curve_rhc_2 = mlrose_hiive.random_hill_climb(problem, restarts = 4, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
_, _, fitness_curve_rhc_3 = mlrose_hiive.random_hill_climb(problem, restarts = 8, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
_, _, fitness_curve_rhc_4 = mlrose_hiive.random_hill_climb(problem, restarts = 12, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
_, _, fitness_curve_rhc_5 = mlrose_hiive.random_hill_climb(problem, restarts = 16, max_attempts = 10, max_iters = 2000, init_state = init_state, curve = True)
print("Completed RHC hyper-parameter testing!")

plt.figure()
plt.plot(fitness_curve_rhc_1[:,0], label = 'restart 0')
plt.plot(fitness_curve_rhc_2[:,0], label = 'restart 4')
plt.plot(fitness_curve_rhc_3[:,0], label = 'restart 8')
plt.plot(fitness_curve_rhc_4[:,0], label = 'restart 12')
plt.plot(fitness_curve_rhc_5[:,0], label = 'restart 16')
plt.title('RHC (FF)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.savefig('FF RHC.png')

_, _, fitness_curve_mimic_1 = mlrose_hiive.mimic(problem, keep_pct = 0.1, pop_size = 100, max_attempts = 10, curve = True)
_, _, fitness_curve_mimic_2 = mlrose_hiive.mimic(problem, keep_pct = 0.3, pop_size = 100, max_attempts = 10, curve = True)
_, _, fitness_curve_mimic_3 = mlrose_hiive.mimic(problem, keep_pct = 0.1, pop_size = 200, max_attempts = 10, curve = True)
_, _, fitness_curve_mimic_4 = mlrose_hiive.mimic(problem, keep_pct = 0.3, pop_size = 200, max_attempts = 10, curve = True)
_, _, fitness_curve_mimic_5 = mlrose_hiive.mimic(problem, keep_pct = 0.1, pop_size = 500, max_attempts = 10, curve = True)
_, _, fitness_curve_mimic_6 = mlrose_hiive.mimic(problem, keep_pct = 0.3, pop_size = 500, max_attempts = 10, curve = True)
print("Completed MIMIC hyper-parameter testing!")

plt.figure()
plt.plot(fitness_curve_mimic_1[:,0], label = 'keep rate 0.1, pop = 100')
plt.plot(fitness_curve_mimic_2[:,0], label = 'keep rate 0.3, pop = 100')
plt.plot(fitness_curve_mimic_3[:,0], label = 'keep rate 0.1, pop = 200')
plt.plot(fitness_curve_mimic_4[:,0], label = 'keep rate 0.3, pop = 200')
plt.plot(fitness_curve_mimic_5[:,0], label = 'keep rate 0.1, pop = 500')
plt.plot(fitness_curve_mimic_6[:,0], label = 'keep rate 0.3, pop = 500')
plt.title('MIMIC (FF)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.savefig('FF MIMIC.png')

_, _, fitness_curve_ga_1 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.1, pop_size = 100, max_attempts = 10, curve = True)
_, _, fitness_curve_ga_2 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.3, pop_size = 100, max_attempts = 10, curve = True)
_, _, fitness_curve_ga_3 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.1, pop_size = 200, max_attempts = 10, curve = True)
_, _, fitness_curve_ga_4 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.3, pop_size = 200, max_attempts = 10, curve = True)
_, _, fitness_curve_ga_5 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.1, pop_size = 500, max_attempts = 10, curve = True)
_, _, fitness_curve_ga_6 = mlrose_hiive.genetic_alg(problem, mutation_prob = 0.3, pop_size = 500, max_attempts = 10, curve = True)
print("Completed GA hyper-parameter testing!")

plt.figure()
plt.plot(fitness_curve_ga_1[:,0], label = 'mutation 0.1, pop = 100')
plt.plot(fitness_curve_ga_2[:,0], label = 'mutation 0.3, pop = 100')
plt.plot(fitness_curve_ga_3[:,0], label = 'mutation 0.1, pop = 200')
plt.plot(fitness_curve_ga_4[:,0], label = 'mutation 0.3, pop = 200')
plt.plot(fitness_curve_ga_5[:,0], label = 'mutation 0.1, pop = 500')
plt.plot(fitness_curve_ga_6[:,0], label = 'mutation 0.3, pop = 500')
plt.title('GA (FF)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.savefig('FF GA.png')