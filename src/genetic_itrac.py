from deap import base
from deap import creator
from deap import tools

import random
import numpy 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import hyperparameter_tuning_genetic_test
import elitism
import itrac_grid
from alpha_vantage.timeseries import TimeSeries

# boundaries for itrac_training parameters:
# "trace_size, width": 1 to 100
# "height": 1 to 100
# "grid_type": 0 to 1
# "operation_type": 0 to 3
# "alpha_plus": 0.0 to 2.0
# "alpha_minus": 0.0 to 2.0
# "eliminate_noise":0 to 0.1


# [n_estimators, learning_rate, algorithm]:
BOUNDS_LOW =  [  1, 1, 0, 0, 0.0, 0.0, 0.0]
BOUNDS_HIGH = [100, 100, 1, 3, 1.0, 1.0, 0.1]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants:
POPULATION_SIZE = 30
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 20
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation



toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# define the hyperparameter attributes individually:
for i in range(NUM_OF_PARAMS):
    # "hyperparameter_0", "hyperparameter_1", ...
    toolbox.register("hyperparameter_" + str(i),
                     random.uniform,
                     BOUNDS_LOW[i],
                     BOUNDS_HIGH[i])

# create a tuple containing an attribute generator for each param searched:
hyperparameters = ()
for i in range(NUM_OF_PARAMS):
    hyperparameters = hyperparameters + \
                      (toolbox.__getattribute__("hyperparameter_" + str(i)),)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator",
                 tools.initCycle,
                 creator.Individual,
                 hyperparameters,
                 n=1)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calculation
def convertParams(individual):
    params = {}
    params['trace_size'] = round(individual[0])
    params['height'] = round(individual[1])
    params['width'] = round(individual[0])
    params['grid_type'] = ['wins', 'earnings'][round(individual[2])]
    params['operation_type'] = ['W-L', 'W/L', 'W/T', 'cW/L'][round(individual[3])]
    params['alpha_plus'] = individual[4]
    params['alpha_minus'] = individual[5]
    params['eliminate_noise_thold'] = individual[6]

    return params



def classificationAccuracy(individual):
    #convert parameters
    trace_size = round(individual[0])
    height = round(individual[1])
    grid_type = ['wins', 'earnings'][round(individual[2])]
    operation_type = ['W-L', 'W/L', 'W/T', 'cW/L'][round(individual[3])]
    alpha_plus = individual[4]
    alpha_minus = individual[5]
    eliminate_noise_thold = individual[6]
    width = trace_size

    if operation_type == 'W-L':
        grid = itrac_grid.Grid(height, width, trace_size, grid_type=grid_type)
    else:
        grid = itrac_grid.ProportionGrid(height, width, trace_size, operation_type=operation_type, grid_type=grid_type)

    info = itrac_grid.performance_last_days(price_vector, 1000, grid, alpha_plus, alpha_minus, eliminate_noise_thold)
    accuracy, participation, acc_above, part_above, acc_below, part_below = info
    #criteria = min(5*0.01*part_below,1)*(2*0.01*acc_below-1)
    criteria = min(5*0.01*part_above,1)*(2*0.01*acc_above-1)
    #criteria = min(5*0.01*participation,1)*(2*0.01*accuracy-1)
    if numpy.isnan(criteria):
        criteria = -1

    return criteria,

toolbox.register("evaluate", classificationAccuracy)

# genetic operators:mutFlipBit

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUNDS_LOW,
                 up=BOUNDS_HIGH,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # print best solution found:
    print("- Best solution is: ")
    print("params = ", convertParams(hof.items[0]))
    print("criteria = %1.5f" % hof.items[0].fitness.values[0])
    print("if you invest 10000, you get = %1.5f" % (hof.items[0].fitness.values[0]*30*10000*0.008))

    #print rest of the info
    def printInfo(individual):
        trace_size = round(individual[0])
        height = round(individual[1])
        grid_type = ['wins', 'earnings'][round(individual[2])]
        operation_type = ['W-L', 'W/L', 'W/T', 'cW/L'][round(individual[3])]
        alpha_plus = individual[4]
        alpha_minus = individual[5]
        eliminate_noise_thold = individual[6]      
        width = trace_size

        if operation_type == 'W-L':
            grid = itrac_grid.Grid(height, width, trace_size, grid_type=grid_type)
        else:
            grid = itrac_grid.ProportionGrid(height, width, trace_size, operation_type=operation_type, grid_type=grid_type)

        info = itrac_grid.performance_last_days(price_vector, 1000, grid, alpha_plus, alpha_minus, eliminate_noise_thold)
        print("accuracy = {}, participation = {}, acc_above = {}, part_above = {}, acc_below = {}, part_below = {}".format(*info))

    printInfo(hof.items[0])

    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.savefig('generations_performance_plot.png')
    plt.show()


if __name__ == "__main__":
    API_KEY = 'MDT9LRDR9TIZGJLH'
    stock = 'SPY'
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    close = data['4. close'].to_numpy()
    price_vector = close[365:]


    main()