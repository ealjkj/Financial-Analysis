import numpy as np

#not used
def split(close):
    """
    Splits data for time series
    """
    train_number = (len(close)-1)*7//10
    validation_number = (len(close)-1)*15//100
    test_number = len(close)-1 - train_number - validation_number
    print(train_number, validation_number, test_number)
    print(train_number+ validation_number+ test_number)

    a = np.split(close, [test_number, validation_number+test_number+1])

    return a

# Historic Wining chance 
def history(price_vector):
    return 100*(sum(price_vector[:-1] - price_vector[1:] > 0)/(len(price_vector)-1))

def get_returns(price_vector):
    return price_vector[:-1]/price_vector[1:]

def labeled_traces_from_array(price_vector, trace_size):
    num_of_traces = len(price_vector)-trace_size
    labeled_traces = []
    for idx in range(1,num_of_traces+1):
        y = price_vector[idx:trace_size+idx]
        y = np.flip(y,0) # The first element on data is the last day. We have to flip the array
        y, label = y/y[-1]-1, price_vector[idx-1]/y[-1]-1
        x = np.linspace(0,trace_size,len(y)) 
        labeled_traces.append([y, label])
    return labeled_traces