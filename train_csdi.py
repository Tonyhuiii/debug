from imputers.CSDI import CSDIImputer
import numpy as np 
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

imputer = CSDIImputer()
data=tf.constant(np.load('datasets/train_ptbxl_1000.npy'))
data=tf.reshape(data, [-1,12,250,4])
data=tf.reshape(data, [-1,250,12])
# print(data.device)
masking='rm'
missing_ratio=0.2
batch= 16

imputer.train(data, masking, missing_ratio, batch_size=batch, path_save='csdi_results/') # for training

# imputer.load_weights('path_to_model', 'path_to_config') # after training

# imputations = imputer.impute(data, mask, number_of_samples) # sampling