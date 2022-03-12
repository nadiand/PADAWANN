import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2
import math

class ClassifyEnv(gym.Env):

  def __init__(self, trainSet, target, secondTrainSet, secondTarget):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you want to use them (we didn't)
    self.batch   = 1000 # Number of images per batch
    self.seed()
    self.viewer = None

    self.trainSet = trainSet
    self.target   = target

    nInputs = np.shape(trainSet)[1]
    high = np.array([1.0]*nInputs)
    self.action_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))

    self.state = None
    self.trainOrder = None
    self.currIndx = None

    # ------------------------------------------------
    # Additional attributes for using a second dataset
    self.t2 = 0
    self.secondTrainSet = secondTrainSet
    self.secondTarget = secondTarget
    self.trainOrder2 = None
    self.currIndx2 = None
    self.state2 = None
    # ------------------------------------------------

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.trainOrder = np.random.permutation(len(self.target))
    self.t = 0 # timestep
    self.currIndx = self.trainOrder[self.t:self.t+self.batch]
    self.state = self.trainSet[self.currIndx,:]

    # ------------------------------------------------
    # Resseting the attributes for the second datset
    self.trainOrder2 = np.random.permutation(len(self.secondTarget))
    self.t2 = 0 # timestep
    self.currIndx2 = self.trainOrder2[self.t2:self.t2+self.batch]
    self.state2 = self.secondTrainSet[self.currIndx2,:]
    # ------------------------------------------------

    return self.state, self.state2
  
  def step(self, action, prediction1, prediction2):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    y = self.target[self.currIndx]
    m = y.shape[0]
    # ------------------------------------------------
    # Calculating the accuracy on the first dataset
    accuracy = 0
    for pred, label in zip(prediction1, y):
      if pred == label:
        accuracy += 1
    accuracy = accuracy/len(y)
    # ------------------------------------------------

    log_likelihood = -np.log(action[range(m),y])
    loss = np.sum(log_likelihood) / m
    reward = -loss

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True
      self.currIndx = self.trainOrder[(self.t*self.batch):\
                                      (self.t*self.batch + self.batch)]

      self.state = self.trainSet[self.currIndx,:]
    else:
      done = True

    # ------------------------------------------------
    # Calculating the accuracy on the second dataset
    accuracy2 = 0
    y2 = self.secondTarget[self.currIndx2]
    m2 = y2.shape[0]
    for pred, label in zip(prediction2, y2):
      if pred == label:
        accuracy2 += 1
    accuracy2 = accuracy2/len(y2)

    # Calculating the loss on the second dataset
    log_likelihood = -np.log(action[range(m2),y2])
    loss = np.sum(log_likelihood) / m2
    reward -= loss

    # Updating based on the loss from second dataset
    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t2 += 1
      done = False
      if self.t2 >= self.t_limit:
        done = True
      self.currIndx2 = self.trainOrder2[(self.t2*self.batch):\
                                      (self.t2*self.batch + self.batch)]

      self.state2 = self.secondTrainSet[self.currIndx2,:]
    else:
      done = True
    # ------------------------------------------------

    obs = self.state
    # Returning the accuracies as well as everything else
    return obs, reward, done, {}, accuracy, accuracy2


# -- Data Sets ----------------------------------------------------------- -- #

def digit_raw():
  ''' 
  Converts 8x8 scikit digits to 
  [samples x pixels]  ([N X 64])
  '''  
  from sklearn import datasets
  digits = datasets.load_digits()
  z = (digits.images/16)
  z = z.reshape(-1, (64))
  return z, digits.target

def mnist_256():
  ''' 
  Converts 28x28 mnist digits to [16x16] 
  [samples x pixels]  ([N X 256])
  '''  
  import mnist
  z = (mnist.train_images()/255)
  z = preprocess(z,(16,16))

  z = z.reshape(-1, (256))
  return z, mnist.train_labels()

# ------------------------------------------------
# Loading the EMNIST 10Letters dataset
def emnist():
  from emnist import extract_training_samples
  images, labels = extract_training_samples('letters')
  
  boolarray = []
  for l in labels:
    if l >=1 and l <=10:
      boolarray.append(True)
    else:
      boolarray.append(False)

  images = images[boolarray]
  z = images/255
  labels = labels[boolarray]
  z = preprocess(z,(16,16))
  z = z.reshape(-1, (256))
  return z, np.array([l-1 for l in labels])
# ------------------------------------------------

def preprocess(img,size, patchCorner=(0,0), patchDim=None, unskew=True):
  """
  Resizes, crops, and unskewes images

  """
  if patchDim == None: patchDim = size
  nImg = np.shape(img)[0]
  procImg  = np.empty((nImg,size[0],size[1]))

  # Unskew and Resize
  if unskew == True:    
    for i in range(nImg):
      procImg[i,:,:] = deskew(cv2.resize(img[i,:,:],size),size)

  # Crop
  cropImg  = np.empty((nImg,patchDim[0],patchDim[1]))
  for i in range(nImg):
    cropImg[i,:,:] = procImg[i,patchCorner[0]:patchCorner[0]+patchDim[0],\
                               patchCorner[1]:patchCorner[1]+patchDim[1]]
  procImg = cropImg

  return procImg

def deskew(image, image_shape, negated=True):
  """
  This method deskwes an image using moments
  :param image: a numpy nd array input image
  :param image_shape: a tuple denoting the image`s shape
  :param negated: a boolean flag telling whether the input image is negated

  :returns: a numpy nd array deskewd image

  source: https://github.com/vsvinayak/mnist-helper
  """
  
  # negate the image
  if not negated:
      image = 255-image
  # calculate the moments of the image
  m = cv2.moments(image)
  if abs(m['mu02']) < 1e-2:
      return image.copy()
  # caclulating the skew
  skew = m['mu11']/m['mu02']
  M = np.float32([[1, skew, -0.5*image_shape[0]*skew], [0,1,0]])
  img = cv2.warpAffine(image, M, image_shape, \
    flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)  
  return img