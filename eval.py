import random
from TetrisSIE import TetrisEnv, condensed_print, gen_scoring_function
from params import get_board_info
import numpy as np
from Visor import BoardVision
import time

def print_stats(use_visuals_in_trace_p, states_p, pieces_p, sleep_time_p):
    vision = BoardVision()
    if use_visuals_in_trace_p:

        for state, piece in zip(states_p, pieces_p):
            vision.update_board(state)
            # print("piece")
            # condensed_print(piece)
            # print('-----')
            time.sleep(sleep_time_p)
        time.sleep(2)
        vision.close()
    else:
        for state, piece in zip(states_p, pieces_p):
            print("board")
            condensed_print(state)
            print("piece")
            condensed_print(piece)
            print('-----')


# Define your fitness function
def fitness_function(env, population):
    fitness_scores = []
    for chromosome in population:
        total_score, states, msg = env.run(gen_scoring_function, chromosome, 500, False)
        fitness_scores.append(total_score)
    return fitness_scores


# Complete the Genetic class
class Genetic:
    def __init__(self, sample_space, pop_size, chrom_len, mutation_rate=0.3, crossover_rate=0.5):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.sample_space = sample_space
        self.chrom_len = chrom_len
        self.population = self._generate_population(pop_size, chrom_len, sample_space)
        self.X, self.Y = self._divide(self.population)

    def _generate_chromosome(self, length, sample):
        # Generates a random chromosome by sampling from the provided sample space
        chromosome = np.random.choice(sample, size=length, replace=False)
        return sample[chromosome]

    def _generate_population(self, pop_size, chrom_size, sample_space):
        # Generates a population by randomly sampling from the sample space
        population = np.random.choice(sample_space, size=(pop_size, chrom_size), replace=True)
        return population

    def _divide(self, population):
        # Divides the population into two halves randomly
        idx = np.arange(len(population))
        np.random.shuffle(idx)
        x_idx = idx[:len(population) // 2]
        y_idx = idx[len(population) // 2:]
        x = population[x_idx]
        y = population[y_idx]
        return x, y

    def _generate_new_generation(self, x_chroms, y_chroms):
        # Generates a new generation by performing crossover between chromosomes
        indices_x = np.random.choice(len(x_chroms), size=len(x_chroms), replace=False)
        x = x_chroms[indices_x]
        indices_y = np.random.choice(len(y_chroms), size=len(y_chroms), replace=False)
        y = y_chroms[indices_y]
        new_x, new_y = self._crossover(x, y)
        return new_x, new_y

    def _crossover(self, x, y):
        # Performs crossover between chromosomes x and y by swapping a portion of their genetic material
        shape = x.shape
        dum = x[:, :shape[1] // 2]
        x[:, :shape[1] // 2] = y[:, :shape[1] // 2]
        y[:, :shape[1] // 2] = dum
        return x, y

    def _fitness_rank(self, chroms, env):
        # Ranks the chromosomes based on their fitness scores using the fitness function
        fitness_scores = fitness_function(env, chroms)
        indices = np.argsort(fitness_scores)
        sorted_chrom = chroms[indices]
        return sorted_chrom

    def _mutate(self, chroms):
        # Performs mutation on the chromosomes by randomly changing a portion of their genetic material
        shape = chroms.shape
        mutation_indices = np.random.choice(np.arange(shape[1]), size=(shape[0], int(shape[0] * self.mutation_rate)), replace=True)
        mutation_values = np.random.uniform(-2, 2, size=(shape[0], int(shape[0] * self.mutation_rate)))
        mutated_chroms = chroms.copy()
        mutated_chroms[np.arange(shape[0])[:, np.newaxis], mutation_indices] += mutation_values
        return mutated_chroms

    def run(self, env, num_generations, num_mutation, mu_gen, epochs_per_generation):
        for i in range(1, num_generations + 1):
            for j in range(epochs_per_generation):
                print(f"Generation {i} epoch {j+1}")
                self.X = self._fitness_rank(self.X, env)
                self.Y = self._fitness_rank(self.Y, env)
                x_rate = int(len(self.X) * self.crossover_rate)
                y_rate = int(len(self.Y) * self.crossover_rate)
                x, y = self._generate_new_generation(self.X[:x_rate], self.Y[:y_rate])
                self.X = np.concatenate((self.X[x_rate:], x), axis=0)
                self.Y = np.concatenate((self.Y[y_rate:], y), axis=0)
            if i % mu_gen == 0 and num_mutation > 0:
                num_mutation -= 1
                self.X = self._mutate(self.X)
                self.Y = self._mutate(self.Y)
                self.X = self._fitness_rank(self.X, env)
                self.Y = self._fitness_rank(self.Y, env)
        return self.X[0], self.Y[0]


# Main code
if __name__ == "__main__":
    # Set up your environment and parameters
    env = TetrisEnv()
    population_size = 25
    num_generations = 4
    mutation_rate = 0.3
    use_visuals_in_trace = True
    sleep_time = 0.1
    chrom_len = 8
    crossover_rate = 0.3
    num_mutation = 3
    mu_gen = 2
    epochs_per_generation = 10
    env.set_seed(39)
    sample_space  = np.random.uniform(-100, 0, size=chrom_len)

    # Create an instance of the Genetic class
    genetic = Genetic(sample_space, population_size, chrom_len, mutation_rate, crossover_rate)

    # Run the genetic algorithm
    final_population = genetic.run(env ,num_generations, num_mutation, mu_gen, epochs_per_generation)

    #Print the final population
    print(f"Final population: {final_population}")
    best_1 = [-65.74792661, -34.75471346, -27.49264635, -34.75471346,
       -78.82680045, -65.77022113, -35.54476263,  -9.71014655]
    best_2= [-66.65968282, -34.75471346, -27.49264635, -34.75471346,
       -76.13964912, -78.40617174, -78.58290884, -64.79227041]


    total_score, states, rate_rot, pieces, msg = env.run(
        gen_scoring_function, best_1, 500, True)
    
    # after running your iterations (which should be at least 500 for each chromosome)
    # you can evolve your new chromosomes from the best after you test all chromosomes here
    print("Ratings and rotations")
    for rr in rate_rot:
        print(rr)
    print(len(rate_rot))
    print('----')
    print(total_score)
    print(msg)
    print_stats(use_visuals_in_trace, states, pieces, sleep_time)

    total_score, states, rate_rot, pieces, msg = env.run(
        gen_scoring_function, best_2, 500, True)
    
    # after running your iterations (which should be at least 500 for each chromosome)
    # you can evolve your new chromosomes from the best after you test all chromosomes here
    print("Ratings and rotations")
    for rr in rate_rot:
        print(rr)
    print(len(rate_rot))
    print('----')
    print(total_score)
    print(msg)
    print_stats(use_visuals_in_trace, states, pieces, sleep_time)


