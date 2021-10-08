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
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
from sklearn.model_selection import TimeSeriesSplit

# boundaries for itrac_training parameters:
# "trace_size, width": 5 to 40
# "height": 1 to 100
# "eliminate_noise":0 to 0.1


# [n_estimators, learning_rate, algorithm]:
BOUNDS_LOW =  [10, 1, 0.0]
BOUNDS_HIGH = [25, 100, 0.01]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 15
HALL_OF_FAME_SIZE = 3
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation
METRIC_TO_OPTIMIZE = 'criteria10'
OPERATOR = lambda x: x

print('-'*100)
print('OPERATOR:', OPERATOR)
toolbox = base.Toolbox()

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))

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
    params['grid_type'] = 'wins'
    params['operation_type'] = 'W-L'
    params['alpha_plus'] = 1
    params['alpha_minus'] = 1
    params['eliminate_noise_thold'] = individual[2]

    return params



def classificationAccuracy(individual):
    #convert parameters
    trace_size = round(individual[0])
    height = round(individual[1])
    grid_type = 'wins'
    operation_type = 'W-L'
    alpha_plus = 1
    alpha_minus = 1
    eliminate_noise_thold = individual[2]
    width = trace_size

    if operation_type == 'W-L':
        grid = itrac_grid.Grid(height, width, trace_size, grid_type=grid_type)
    else:
        grid = itrac_grid.ProportionGrid(height, width, trace_size, operation_type=operation_type, grid_type=grid_type)

    dataset = labeled_traces_from_array(all_days, trace_size, operator=OPERATOR)

    X = np.array([trace for (trace, _) in dataset])
    y = np.array([int(label >= 0) for (_, label) in dataset])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23, test_size=0.2, stratify=y)

    metric_sum = 0
    val_acc_sum = 0
    for train_index, test_index in kf.split(X_train):
        kX_train, kX_test = X_train[train_index], X_train[test_index]
        ky_train, ky_test = y_train[train_index], y_train[test_index]
        
        grid.fit(kX_train, ky_train, eliminate_noise_thold=eliminate_noise_thold)
        metrics = grid.get_metrics(kX_test, ky_test, alpha_plus=alpha_plus, alpha_minus=alpha_minus)
        metric = metrics[METRIC_TO_OPTIMIZE]
        val_acc = metrics['accuracy']
        
        metric_sum += metric
        val_acc_sum += val_acc 

         

        
    average_metric = metric_sum/n_splits
    average_acc = val_acc_sum/n_splits
    individual.val_acc = average_acc


    if numpy.isnan(average_metric):
        average_metric = -1

    ans = average_metric, 
    return ans
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
def main(stock):

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
        grid_type = 'wins'
        operation_type = 'W/L'
        alpha_plus = 1
        alpha_minus = 1
        eliminate_noise_thold = individual[2]      
        width = trace_size

        if operation_type == 'W-L':
            grid = itrac_grid.Grid(height, width, trace_size, grid_type=grid_type)
        else:
            grid = itrac_grid.ProportionGrid(height, width, trace_size, operation_type=operation_type, grid_type=grid_type)

        price_vector = close
        info = itrac_grid.performance_last_days(price_vector, TEST_SIZE, grid, alpha_plus, alpha_minus, eliminate_noise_thold)
        print('-'*80)
        print('Last' + str(TEST_SIZE) + ' days report')
        print("accuracy = {}, participation = {}, acc_above = {}, part_above = {}, acc_below = {}, part_below = {}".format(*info))
        criteria10_above = min(1, 0.01*info[3]*10)*(2*info[2]*0.01-1)
        print(f"criteria10_above = {criteria10_above}")

        return info


    last_n_days_info = printInfo(hof.items[0]) 
    individual_val_acc = hof.items[0].val_acc


    to_export = {}
    for key in convertParams(hof.items[0]):
        to_export[key] = convertParams(hof.items[0])[key]
    
    to_export["test_accuracy"] = last_n_days_info[0]
    to_export["test_participation"] = last_n_days_info[1]
    to_export["test_acc_above"] = last_n_days_info[2]
    to_export["test_part_above"] = last_n_days_info[3]
    to_export["test_acc_below"] = last_n_days_info[4]
    to_export["test_part_below"] = last_n_days_info[5]


    to_export["val_"+ METRIC_TO_OPTIMIZE] = hof.items[0].fitness.values[0]
    to_export["val_acc"] = individual_val_acc
    
    print("individual_val_acc = ", individual_val_acc)
    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    with open("improved_params/" +stock + ".json", "w") as file:
        json.dump(to_export, file,  indent=4)
    
    # # plot statistics:
    # sns.set_style("whitegrid")
    # plt.plot(maxFitnessValues, color='red')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Generation')
    # plt.ylabel('Max / Average Fitness')
    # plt.title('Max and Average fitness over Generations')
    # plt.savefig('generations_performance_plot.png')
    # plt.show()

    return to_export

if __name__ == "__main__":
    API_KEY = 'MDT9LRDR9TIZGJLH'
    TEST_SIZE = 365
    # stock = 'AEHR'
    


    

    n_splits = 4
    # kf = KFold(n_splits=n_splits)
    kf = TimeSeriesSplit(n_splits)

    symbol_list = ['SPY', 'ACU', 'AHPI', 'ALOT', 'AMHC', 'BSQR', 'CLWT', 'CSPI', 'CTIB', 'DAIO', 'DGICB', 'EDAP', 'ELTK', 'FONR', 'HSKA', 'ICCC', 'IMH', 'INOD', 'INS', 'ISIG', 'KTCC', 'LAKE', 'LPTH', 'MGF', 'MSON', 'NATH', 'NOM']

    for stock in symbol_list:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
        close = data['4. close'].to_numpy()
        all_days = close[TEST_SIZE:]

        main(stock)