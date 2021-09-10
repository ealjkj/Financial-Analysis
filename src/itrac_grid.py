import pandas as pd 
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time  
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns
from scipy import interpolate

class Grid():
  def __init__(self, height, width, trace_size, grid_type = 'wins', interpolation=False):
    self.trace_size = trace_size
    self.grid_type = grid_type
    self.epsilon = 0
    self.height = height
    self.width = width
    self.temp_values = np.ones((height,width,2))*self.epsilon
    self.mean = None
    self.std = None
    self.train_traces = []
    self.operation_type='W-L'
    self.interpolation = interpolation

  def feed_one_trace(self, x, y, label, max_h=0.02, min_h=-0.12, max_w=60, min_w=0,eliminate_noise_thold=0, debug = False):
    if np.abs(label) >= eliminate_noise_thold:
      h_step = (max_h - min_h)/self.height
      w_step = (max_w - min_w)/self.width
      for i,f_i in zip(x, y):
        x_interval = int((i-min_w)//w_step)
        y_interval = int((f_i-min_h)//h_step)
        try:
          if label >0:
            if self.grid_type == 'wins':
              self.temp_values[y_interval, x_interval] = np.array([1,0])
            elif self.grid_type == 'earnings':
              self.temp_values[y_interval, x_interval] = np.array([label,0])
          else:
            if self.grid_type == 'wins':
              self.temp_values[y_interval, x_interval] = np.array([0,1])
            elif self.grid_type == 'earnings':
              self.temp_values[y_interval, x_interval] = np.array([0,-label])
        except:
          pass
          if True:
            print(f'--There is a problem at the {i,f_i} iteration1 | ',y_interval, x_interval)
            print(max_h, min_h, max_h-min_h, (max_h-min_h)/self.height, h_step, w_step)
#           else:
#             print('There is a problem. Please activate the debug parameter to know why')
#             break
#             break
#         else:
#           pass
  def pre_train(self, close, eliminate_noise_thold, debug = False):
    self.train_vector = close
    num_of_traces_grid = len(close)-self.trace_size
    self.train_set = close.tolist()
    self.temp_values = np.ones((self.height,self.width, 2))*self.epsilon
    temp_grid = Grid(self.height, self.width, self.trace_size, grid_type=self.grid_type)
    padding = 0.05

    # Mins and max
    self.max_h = max([max(close[idx:self.trace_size+idx]/close[idx:self.trace_size+idx][0]-1) for idx in range(1,num_of_traces_grid+1)])+padding
    self.min_h = min([min(close[idx:self.trace_size+idx]/close[idx:self.trace_size+idx][0]-1) for idx in range(1,num_of_traces_grid+1)])-padding
    self.min_w = 0
    self.max_w = self.trace_size

    
    if debug:
      print(self.max_h, self.min_h)

    for idx in range(1,num_of_traces_grid+1):
      y = close[idx:self.trace_size+idx]
      y = np.flip(y,0) # The first element on data is the last day. We have to flip the array
      y, label = y/y[-1]-1, close[idx-1]/y[-1]-1
      x = np.linspace(0,self.trace_size-1,len(y))

      if self.interpolation:
        num_of_points = self.width
        splines = interpolate.splrep(x,y)
        #now redefine x and y
        x = np.linspace(0,self.trace_size-1, num_of_points)
        y = interpolate.splev(x,splines)

      
      self.train_traces.append((y,label))
      temp_grid.feed_one_trace(x,y,label,max_h = self.max_h, min_h = self.min_h, max_w = self.max_w, eliminate_noise_thold=eliminate_noise_thold, debug = debug)
      self.temp_values+=temp_grid.temp_values

  def fit(self, X,y, eliminate_noise_thold=0):
    padding = 0.05
    self.temp_values = np.ones((self.height,self.width, 2))*self.epsilon
    self.train_traces = list(zip(X,y))
    # Mins and max
    self.max_h = max([max(trace) for trace in X])+padding
    self.min_h = min([min(trace) for trace in X]) -padding
    self.min_w = 0
    self.max_w = self.trace_size
    

    for trace, label in zip(X,y):
      x = np.linspace(0,self.trace_size-1,len(trace))
      temp_grid = Grid(self.height, self.width, self.trace_size, grid_type=self.grid_type)
      temp_grid.feed_one_trace(x,trace,label,max_h = self.max_h, min_h = self.min_h, max_w = self.max_w, eliminate_noise_thold=eliminate_noise_thold)
      self.temp_values+=temp_grid.temp_values
      
    self.make_operation()
      
      
      

                     
        
    
    
  def make_operation(self):
    self.values = self.temp_values[:,:,0]-self.temp_values[:,:,1]

  def train(self, price_vector, eliminate_noise_thold = 0, debug = False):
    self.pre_train(price_vector, eliminate_noise_thold=eliminate_noise_thold, debug = False)
    self.make_operation()

  def update(self, x, y, label):
    temp_grid = Grid(self.height, self.width, self.trace_size, grid_type=self.grid_type)
    temp_grid.feed_one_trace(x,y, label, max_h=self.max_h, min_h=self.min_h, max_w=self.max_w, min_w=self.min_w)

    #Update temp_values
    self.temp_values+=temp_grid.temp_values
    self.make_operation()

    #update train_vector
    self.train_traces.append((y, label))
      
    
  def binary_image(self, y_reverse = False, resize = None):
    # Plot the grid
    img = Image.new(mode='RGB', size=(self.width,self.height), color=(0, 0, 255))
    img = np.array(img)
    img.setflags(write=1)

    for i in range(self.values.shape[0]):
      for j in range(self.values.shape[1]):
        if self.values[i,j] > 0:
          img[i,j] = np.array([0,255,0])
        elif self.values[i,j] < 0:
          img[i,j] = np.array([255,0,0])
        else:
          img[i,j] = np.array([0,0,0])

    img = Image.fromarray(img)
    if y_reverse:
      img = ImageOps.flip(img)

    if resize is not None:
      img = img.resize(size=resize)

    return img

  def evaluate(self, x, y):
    out_of_limits_counter = 0
    if self.interpolation:
      num_of_points = self.width
      splines = interpolate.splrep(x,y)
      #now redefine x and y
      x = np.linspace(0,self.trace_size-1, num_of_points)
      y = interpolate.splev(x,splines)
      
    h_step = (self.max_h - self.min_h)/self.height
    w_step = (self.max_w - self.min_w)/self.width
    score = 0
    for i,f_i in zip(x, y):
      x_interval = int((i-self.min_w)//w_step)
      y_interval = int((f_i-self.min_h)//h_step)
      try:
        score += self.values[y_interval, x_interval]
      except:
        out_of_limits_counter+=1
#         print(f'values = {self.values}')
        # print(f'{y_interval} should be between  {0} and {self.height:.2f} \n{x_interval} should be between {0} and {self.width:.2f}\n' )
    if out_of_limits_counter > 0:
      pass
#       print(out_of_limits_counter, 'values were ignored')

      
    return score

  def evaluate2(self, x, y):
    h_step = (self.max_h - self.min_h)/self.height
    w_step = (self.max_w - self.min_w)/self.width
    set_of_coordinates = set()
    for i,f_i in zip(x, y):
      x_interval = int((i-self.min_w)//w_step)
      y_interval = int((f_i-self.min_h)//h_step)
      set_of_coordinates.add((x_interval, y_interval))
    score = 0
    for x_interval, y_interval in list(set_of_coordinates):
      score += self.values[y_interval, x_interval]
    return score

  def predictions(self, price_vector):
    scores = []
    first_debug = True
    for idx in range(len(price_vector)-self.trace_size):
      y = price_vector[idx:self.trace_size+idx]
      y = np.flip(y,0) # The first element on data is the last day. We have to flip the array
      y, label = y/y[-1]-1, price_vector[idx-1]/y[-1]-1
      x = np.linspace(0,self.max_w-1,len(y))

      scores.append((self.evaluate(x,y), idx, label > 0))
    return scores

  def predictions_from_ltraces(self, labeled_traces):
    scores = []
    for idx, (trace, label) in enumerate(labeled_traces):
      x = np.linspace(0,self.max_w-1,len(trace))
      scores.append((self.evaluate(x,trace), idx, label>0))
    return scores

  def get_mean(self):
    if self.mean is None or True:
      scores = np.array(self.predictions_from_ltraces(self.train_traces))
      scores = scores[:,0]
      self.mean = np.mean(scores)
    return self.mean

  def get_std(self):
    if self.std is None or True:
      scores = np.array(self.predictions_from_ltraces(self.train_traces))[:,0]
      self.std = np.std(scores)
    return self.std

  def get_signal(self, trace, up_thold, down_thold):
    x = np.linspace(0, self.trace_size-1, self.trace_size)
    if self.interpolation:
      num_of_points = self.width 
      splines = interpolate.splrep(x,trace)
      #now redefine x and y
      x = np.linspace(0,self.trace_size-1, num_of_points)
      trace = interpolate.splev(x,splines)
    score = self.evaluate(x, trace)
    ans = 'stay'
    if score >= up_thold:
      ans = 'buy'
    elif score <= down_thold:
      ans = 'sell'
    return ans

  def get_metrics(self, X, y, alpha_plus, alpha_minus):
    metrics, scores, pos = {}, [], []
    metrics['mean'] = self.get_mean()
    metrics['std'] = self.get_std()
    metrics['up_thold'] = metrics['mean'] + alpha_plus*metrics['std']
    metrics['down_thold'] = metrics['mean'] -alpha_minus*metrics['std']
    
    for trace, label in zip(X,y):
      x = np.linspace(0,self.max_w-1,len(trace))
      scores.append(self.evaluate(x,trace))
      pos.append(label>0)
      
    scores, pos = np.array(scores), np.array(pos)
    mask0 = pos == 0
    mask1 = pos == 1

    red = scores[mask0]
    green = scores[mask1]
    
    correct_red = sum(red < metrics['down_thold'])
    incorrect_red = sum(red > metrics['up_thold'])
    correct_green = sum(green > metrics['up_thold'])
    incorrect_green = sum(green < metrics['down_thold'])
    number_of_predictions = correct_green+correct_red + incorrect_red + incorrect_green
    total = len(green) + len(red)
    
#     print(f'number of predictions: {number_of_predictions}, correct_red: {correct_red}, correct green: {correct_green}, incorrect red {incorrect_red}, incorrect green: {incorrect_green}, total: {total}')
    metrics['accuracy'] = 100*(correct_red + correct_green)/number_of_predictions

    metrics['acc_above'] = 100*correct_green/(correct_green + incorrect_red)
    metrics['acc_below'] = 100*correct_red/(correct_red + incorrect_green)
    metrics['part_above'] = 100*(incorrect_red + correct_green)/total
    metrics['part_below'] = 100*(incorrect_green + correct_red)/total
    metrics['participation'] = 100*number_of_predictions/total
    metrics['criteria'] = (2*0.01*metrics['accuracy'] - 1)*0.01*metrics['participation']
    metrics['criteria2'] = (2*0.01*metrics['accuracy'] - 1)*min(1,5*0.01*metrics['participation'])
    metrics['criteria_above'] = (2*0.01*metrics['acc_above'] - 1)*0.01*metrics['part_above']
    metrics['criteria2_above'] = (2*0.01*metrics['acc_above'] - 1)*min(1,5*0.01*metrics['part_above'])
    metrics['criteria_below'] = (2*0.01*metrics['acc_below'] - 1)*0.01*metrics['part_below']
    metrics['criteria2_below'] = (2*0.01*metrics['acc_below'] - 1)*min(1,5*0.01*metrics['part_below'])
    return metrics


class ProportionGrid(Grid):
  def __init__(self, height, width, trace_size, grid_type = 'wins', operation_type = 'W/L', interpolation=False):
    self.grid_type = grid_type
    self.trace_size = trace_size
    self.operation_type = operation_type
    self.epsilon = 1000
    self.height = height
    self.width = width
    self.temp_values = np.ones((height,width,2))*self.epsilon
    self.mean = None
    self.std = None
    self.train_traces = []
    self.interpolation = interpolation

  def make_operation(self):
    #Wins/Losses
    if self.operation_type == 'W/L':
      self.values = self.temp_values[:,:,0]/self.temp_values[:,:,1]
    elif self.operation_type == 'W/T':
      self.values = self.temp_values[:,:,0]/(self.temp_values[:,:,1]+self.temp_values[:,:,0])
    elif self.operation_type == 'T/L':
      self.values = 1/(self.temp_values[:,:,1]/(self.temp_values[:,:,1]+self.temp_values[:,:,0]))

class MultipleGrid():
  def __init__(self, height, width, trace_size, num_of_grids, grid_type = 'wins', operation_type = 'W/L', ponderation=1):
    self.grid_type = grid_type
    self.trace_size = trace_size
    self.operation_type = operation_type
    self.height = height
    self.width = width
    self.train_traces = []
    self.list_of_grids = []
    self.num_of_grids = num_of_grids
    self.ponderation=ponderation
    self.mean = None
    self.std = None
    # Appending grids
    for i in range(self.num_of_grids):
      if operation_type == 'W-L':
        self.list_of_grids.append(Grid(self.height, self.width, self.trace_size, self.grid_type))
      else:
        self.list_of_grids.append(ProportionGrid(self.height, self.width, self.trace_size, self.grid_type, self.operation_type))

  def train(self, price_vector, eliminate_noise_thold=0):
    interval_num_of_elements = int(len(price_vector)/self.num_of_grids)
    for i in range(self.num_of_grids):
      self.list_of_grids[i].train(price_vector[:(i+1)*interval_num_of_elements], eliminate_noise_thold)
    self.train_traces = self.list_of_grids[-1].train_traces

  def evaluate(self, x,y):
    return np.sum(np.array([g.evaluate(x,y) for g in self.list_of_grids])*self.ponderation)

  def predictions_from_ltraces(self, labeled_traces):
    scores = []
    for idx, (trace, label) in enumerate(labeled_traces):
      x = np.linspace(0,self.trace_size-1,len(trace))
      scores.append((self.evaluate(x,trace), idx, label>0))
    return scores

  def get_mean(self):
    if self.mean is None or True:
      print('Computing mean ...')
      scores = np.array(self.predictions_from_ltraces(self.train_traces))[:,0]
      self.mean = np.mean(scores)
    return self.mean

  def get_std(self):
    if self.std is None or True:
      print('Computing std ...')
      scores = np.array(self.predictions_from_ltraces(self.train_traces))[:,0]
      self.std = np.std(scores)
    return self.std


class CorrectGrid(Grid):
  def __init__(self, height, width, trace_size, grid_type = 'wins'):
    self.trace_size = trace_size
    self.grid_type = grid_type
    self.epsilon = 0
    self.height = height
    self.width = width
    self.temp_values = np.ones((height,width,2))*self.epsilon
    self.mean = None
    self.std = None
    self.train_traces = []
    self.operation_type='W-L'

  def feed_one_trace(self, x, y, label, max_h=0.02, min_h=-0.12, max_w=60, min_w=0,eliminate_noise_thold=0, debug = False):
    if np.abs(label) >= eliminate_noise_thold:
      h_step = (max_h - min_h)/self.height
      w_step = (max_w - min_w)/self.width
      for i,f_i in zip(x, y):
        x_interval = int((i-min_w)//w_step)
        y_interval = int((f_i-min_h)//h_step)
        set_of_coordinates.add((x_interval, y_interval))
        try:
          if label >0:
            if self.grid_type == 'wins':
              set_of_cordinates.add((self.temp_values[y_interval, x_interval],np.array([1,0]))) 
            elif self.grid_type == 'earnings':
              set_of_cordinates.add((self.temp_values[y_interval, x_interval],np.array([label,0]))) 
          else:
            if self.grid_type == 'wins':
              set_of_cordinates.add((self.temp_values[y_interval, x_interval],np.array([0,1]))) 
            elif self.grid_type == 'earnings':
              set_of_cordinates.add((self.temp_values[y_interval, x_interval],np.array([0,-label])))  
          for x_interval, y_interval, to_add in list(set_of_coordinates):
            self.temp_values[y_interval, x_interval] = to_add
        except:
          if debug:
            print(f'--There is a problem at the {i,f_i} iteration',y_interval, x_interval)
            print(max_h, min_h, max_h-min_h, (max_h-min_h)/self.height, h_step, w_step)
          else:
            print('There is a problem. Please activate the debug parameter to know why')
            break
            break
        else:
          pass
  def evaluate(self, x, y):
    h_step = (self.max_h - self.min_h)/self.height
    w_step = (self.max_w - self.min_w)/self.width
    set_of_coordinates = set()
    for i,f_i in zip(x, y):
      x_interval = int((i-self.min_w)//w_step)
      y_interval = int((f_i-self.min_h)//h_step)
      set_of_coordinates.add((x_interval, y_interval))
    score = 0
    for x_interval, y_interval in list(set_of_coordinates):
      score += self.values[y_interval, x_interval]
    return score


#-------------------------------------------- FUNCTIONS ---------------------------------------

def last_days_grid_info(price_vector, last_n_days, grid, eliminate_noise_thold=0):
  grid.train(price_vector[last_n_days:], eliminate_noise_thold=eliminate_noise_thold)
  number_of_samples = len(grid.train_vector)
  mean = grid.get_mean()
  std = grid.get_std()


  #Initialize 
  info = []
  for i in range(last_n_days):
    trace = price_vector[last_n_days-i: last_n_days-i+grid.trace_size]
    #print('debug', i, trace, price_vector[last_n_days-i-1])
    trace = np.flip(trace,0) # The first element on data is the last day. We have to flip the array
    trace, label = trace/trace[-1]-1, price_vector[last_n_days-i-1]/trace[-1]-1
    x = np.linspace(0,grid.max_w-1,len(trace))

    info.append((grid.evaluate(x,trace), i, label > 0))
    grid.update(x,trace, label)

  pred = np.array(info)
  return [pred, mean, std]

def performance_data(pred, mean, std, alpha_plus, alpha_minus):
  up_thold = mean + alpha_plus*std
  down_thold = mean - alpha_minus*std

  mask0 = pred[:,2] == 0
  mask1 = pred[:,2] == 1

  red = pred[mask0]
  green = pred[mask1]

  correct_red = sum(red[:,0] < down_thold)
  incorrect_red = sum(red[:,0] > up_thold)
  correct_green = sum(green[:,0] > up_thold)
  incorrect_green = sum(green[:,0] < down_thold)

  #metrics
  number_of_predictions = correct_green+correct_red + incorrect_red + incorrect_green
  acc = 100*(correct_red + correct_green)/number_of_predictions
  total = len(green) + len(red)
  acc_above = 100*correct_green/(correct_green + incorrect_red)
  acc_below = 100*correct_red/(correct_red + incorrect_green)
  part_above = 100*(incorrect_red + correct_green)/total
  part_below = 100*(incorrect_green + correct_red)/total
  part = 100*number_of_predictions/total

  return [acc, part, acc_above, part_above, acc_below, part_below]

def performance_last_days(price_vector, last_n_days, grid, alpha_plus, alpha_minus, eliminate_noise_thold=0):
  grid.train(price_vector[last_n_days:], eliminate_noise_thold=eliminate_noise_thold)
  number_of_samples = len(grid.train_vector)
  mean = grid.get_mean()
  std = grid.get_std()

  up_thold = mean + alpha_plus*std
  down_thold = mean - alpha_minus*std

  #Initialize 
  info = []
  up_tholds = np.zeros(last_n_days)
  down_tholds = np.zeros(last_n_days)

  for i in range(last_n_days):
    trace = price_vector[last_n_days-i: last_n_days-i+grid.trace_size]
    #print('debug', i, trace, price_vector[last_n_days-i-1])
    trace = np.flip(trace,0) # The first element on data is the last day. We have to flip the array
    trace, label = trace/trace[-1]-1, price_vector[last_n_days-i-1]/trace[-1]-1
    x = np.linspace(0,grid.max_w-1,len(trace))

    #Save results
    up_tholds[i] = mean + alpha_plus*std
    down_tholds[i] = mean - alpha_minus*std

    info.append((grid.evaluate(x,trace), i, label > 0))

    grid.update(x,trace, label)

  pred = np.array(info)

  mask0 = pred[:,2] == 0
  mask1 = pred[:,2] == 1

  red = pred[mask0]
  green = pred[mask1]

  correct_red = sum(red[:,0] < down_thold)
  incorrect_red = sum(red[:,0] > up_thold)
  correct_green = sum(green[:,0] > up_thold)
  incorrect_green = sum(green[:,0] < down_thold)

  #metrics
  number_of_predictions = correct_green+correct_red + incorrect_red + incorrect_green
  acc = 100*(correct_red + correct_green)/number_of_predictions
  total = len(green) + len(red)
  acc_above = 100*correct_green/(correct_green + incorrect_red)
  acc_below = 100*correct_red/(correct_red + incorrect_green)
  part_above = 100*(incorrect_red + correct_green)/total
  part_below = 100*(incorrect_green + correct_red)/total
  part = 100*number_of_predictions/total

  #Plot
  #f = plt.figure()
  #f.set_figwidth(20)
  #f.set_figheight(10)

  #plt.plot(pred[:,0], alpha=0.1)
  #plt.plot(up_tholds, '--')
  #plt.plot(down_tholds, '--', color='#333333')
  #plt.scatter(red[:,1], red[:,0], s= 3, c='red', alpha = 1)
  #plt.scatter(green[:,1], green[:,0], s= 3, c='green', alpha=1)

  #title = 'last_' + str(last_n_days) + '_days_'+ str((grid.trace_size, grid.height, grid.width)) +'_'+ grid.grid_type+'_'+grid.operation_type
  #plt.savefig(title)
  
  return [acc, part, acc_above, part_above, acc_below, part_below]
  #return ([acc, part, acc_above, part_above, acc_below, part_below], [pred, up_thold, down_thold])
