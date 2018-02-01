import tensorflow as tf 
import numpy as np 
import pandas as pd 
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import  TensorBoard
from data_maker import *

input_data_size = 3000

data = data(input_data_size)
data = convert_data(data)
