from livelossplot import PlotLosses
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import time
import random
import copy

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True
	
	
class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (Tensor): A tensor containing the data e.g. images
            targets (Tensor): A tensor containing all the labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]
        sample = sample.view(1, 28, 28).float()
        if self.transform:
            sample = self.transform(sample)

        return sample, label

		
def normalise_image(data, mean, std):
  X_norm = data[:].float()
  X_norm = X_norm - mean
  X_norm = X_norm / std
  return X_norm


class SupervisedLearning:
    def __init__(self, X, y, model, optimiser, loss_function, batch_size, test_batch_size,
                 device="cpu", 
                 transform=False, 
                 seed=42, n_epochs=30,
                 val_ratio=0.1, n_splits=1, 
                 early_stop = True,
                 patience = 5,
                 tol = 0.001):
      """
      
      """
      
      self.device = device
      
      self.X = X.float()/255.
      self.y = y
      
      self.model = model.to(self.device)
      self.optimiser = optimiser
      self.loss_function = loss_function
      
      self.X_train = None
      self.X_val = None
      
      self.y_train = None
      self.y_val = None
      
      self.transform = transform
      
      assert(batch_size > 0 and batch_size < int(0.1 * X.size()[0]))
      self.batch_size = batch_size
      assert(test_batch_size > 0 and test_batch_size < int(0.1 * X.size()[0]))
      self.test_batch_size = test_batch_size
      self.n_epochs = n_epochs
      self.seed = seed
      self.val_ratio = val_ratio
      self.n_splits = n_splits

      self.trained_full=False
      
      self.mean_full = None
      self.std_full = None
      
      self.mean = None
      self.std = None
      
      self.logs = None # saves the liveloss object data
      
      self.best_model = None
      
      self.early_stop = early_stop
      if self.early_stop: self.early =  early_stopping(patience=patience, rel_tol=tol)  
      
      
    def split_data(self):
      sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.val_ratio, random_state=0)
      sss.get_n_splits(self.X, self.y)

      for train_index, val_index in sss.split(self.X, self.y):
        self.X_train, self.X_val = self.X[train_index], self.X[val_index]
        self.y_train, self.y_val = self.y[train_index], self.y[val_index]
        
      return None
      
      
      
    def train(self, train_data_loader):
      self.model.train()                # set model to train mode
      
      train_loss, train_accuracy = 0., 0.

      for Xtr, ytr in train_data_loader:# X and y are data inside a batch specified
                                        # at train_data_loader

        Xtr = Xtr.to(self.device)
        ytr = ytr.to(self.device)
        

        self.optimiser.zero_grad()           # reset gradients
        zn = self.model(Xtr)                 # perform forward pass
        

        loss = self.loss_function(zn, ytr)   # compute loss value over batch
        loss.backward()                 # perform backward pass
        train_loss += (loss * Xtr.size()[0]).detach().cpu().numpy()
        

        y_pred = F.log_softmax(zn, dim=1).max(1)[1]
        train_accuracy += accuracy_score(ytr.cpu().numpy(), y_pred.detach().cpu().numpy())*Xtr.size()[0]
        
        self.optimiser.step()               # optimisation step
        
      return train_loss/len(train_data_loader.dataset), train_accuracy/len(train_data_loader.dataset)
    
    
    
    def validate(self, val_data_loader):
      self.model.eval()                     # set model to evaluation mode
      
      validation_loss, validation_accuracy = 0., 0.
      
      for Xv, yv in val_data_loader:
        with torch.no_grad():
          
          Xv, yv = Xv.to(self.device), yv.to(self.device)
          
          zn = self.model(Xv)
          loss = self.loss_function(zn, yv)
          validation_loss += (loss * Xv.size(0)).detach().cpu().numpy()
          
          y_pred = F.log_softmax(zn, dim=1).max(1)[1]
          validation_accuracy += accuracy_score(yv.cpu().numpy(), y_pred.detach().cpu().numpy())*Xv.size(0)

            
      return validation_loss/len(val_data_loader.dataset), validation_accuracy/len(val_data_loader.dataset)
        
      
    
    def train_wrapper(self, train_full=False, plot_loss=True):
      # start timer
      t = time.time()
      
      # set seed
      set_seed(int(self.seed))
      
      
      if train_full: # train with full data (train + validation)
        # find mean and std of training data
        mean, std = self.find_mean_std(train_full)
        
        # create train_transform
        if self.transform:
          train_transform =  transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.RandomRotation(10),
                              transforms.RandomCrop(28, pad_if_needed=True),
                              transforms.ToTensor(), 
                              transforms.Normalize(mean=[mean], std=[std])
                              ])
        else:
          train_transform =  transforms.Compose([
                              transforms.Normalize(mean=[mean], std=[std])
                              ])
        
        # create dataloaders
        train_dataset = CustomTensorDataset(self.X, self.y, transform=train_transform)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        best_train_acc = 0
        
        # train and validate
        if plot_loss: liveloss = PlotLosses()
        for epoch in range(self.n_epochs):
            logs = {}
            train_loss, train_accuracy = self.train(train_data_loader)

            if plot_loss:
              logs['' + 'log loss'] = train_loss.item()
              logs['' + 'accuracy'] = train_accuracy.item()
              logs['val_' + 'log loss'] = 1.
              logs['val_' + 'accuracy'] = 1.
              liveloss.update(logs)
              liveloss.draw()
              logs['time'] = time.time() - t
              
            # Checking stopping criteria
            if self.early_stop: self.early(train_accuracy)
                
            # Saving the best weights    
            if train_accuracy.item() > best_train_acc:  
              # Saving the models weights to best model
              self.best_model = self.model
              best_train_acc = train_accuracy.item()
            
            # If the stopping criteria is met  
            if self.early.stop: 
                self.logs = logs
                self.model = self.best_model
                break
            self.logs = logs  
                  
        self.trained_full=True
        

      else:
        # split data
        self.split_data()
        
        # find mean and std of training data
        mean, std = self.find_mean_std(train_full)
        
        
        # create transforms
        if self.transform:
          train_transform =  transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.RandomRotation(10),
                              transforms.RandomCrop(28, pad_if_needed=True),
                              transforms.ToTensor(), 
                              transforms.Normalize(mean=[mean], std=[std])
                              ])
        else:
          train_transform =  transforms.Compose([
                              transforms.Normalize(mean=[mean], std=[std])
                              ])
            
        val_transform = transforms.Compose([
                            transforms.Normalize(mean=[mean], std=[std])
                            ])

        # create datasets and dataloaders
        train_dataset = CustomTensorDataset(self.X_train, self.y_train, transform=train_transform)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = CustomTensorDataset(self.X_val, self.y_val, transform=val_transform)
        val_data_loader = DataLoader(val_dataset, batch_size=self.test_batch_size, shuffle=False)
        
        best_val_acc = 0
            
        # train and validate
        if plot_loss: liveloss = PlotLosses()
        for epoch in range(self.n_epochs):
            logs = {}
            train_loss, train_accuracy = self.train(train_data_loader)
            val_loss, val_accuracy = self.validate(val_data_loader)
            
            if plot_loss:
              logs['' + 'log loss'] = train_loss.item()
              logs['' + 'accuracy'] = train_accuracy.item()
              logs['val_' + 'log loss'] = val_loss.item()
              logs['val_' + 'accuracy'] = val_accuracy.item() # liveloss wants it plotted
              liveloss.update(logs)
              liveloss.draw()
              logs['time'] = time.time() - t 
            
            # Checking stopping criteria
            if self.early_stop: self.early(val_accuracy)
                
            # Saving the best weights    
            if val_accuracy.item() > best_val_acc:  
              self.best_model = self.model
              best_val_acc = val_accuracy.item()
            
            # If the stopping criteria is met  
            if self.early.stop: 
                self.logs = logs
                self.model = self.best_model
                break

            self.logs = logs
        self.trained_full=False
        self.model = self.best_model
              
      return None
            
            

    def find_mean_std(self, full_training=False):
      """
      Finds the mean and std values of a normalised image (divided by 255)
      """
      mean = 0
      std = 0
      
      if full_training:
        mean = torch.mean(self.X)
        std = torch.std(self.X)
        
        self.mean_full = mean
        self.std_full = std
        
      else:
        mean = torch.mean(self.X_train)
        std = torch.std(self.X_train)
        self.mean = mean
        self.std = std
        
      return mean, std



class KFoldValidation(SupervisedLearning):
    def __init__(self, X, y, model, optimiser, loss_function, batch_size, test_batch_size,
                 device="cpu", 
                 confusion_matrix=True, 
                 transform=True,
                 seed=42, n_epochs=30,
                 n_folds=3, 
                 early_stop = False,
                 patience = 5,
                 tol = 0.001):
        self.device = device
        self.X = X
        self.y = y
        self.model_ori = model.to(self.device)
        self.optimiser_ori = optimiser
        
        self.model = model.to(self.device)
        self.optimiser = optimiser
        self.loss_function = loss_function
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.transform = transform
        #self.train_transform = train_transform
        #self.val_transform = val_transform
        
        
        self.result = None # The cross validation result 
        #assert(batch_size > 0 and batch_size < int(0.1 * X.size()[0]))
        assert(batch_size > 0 and batch_size < int(0.1 * len(X)))
        self.batch_size = batch_size
        #assert(test_batch_size > 0 and test_batch_size < int(0.1 * X.size()[0]))
        assert(test_batch_size > 0 and test_batch_size < int(0.1 * len(X)))
        
        self.test_batch_size = test_batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.n_folds = n_folds
        
        self.mean = None
        self.std = None
        #self.logs = None # saves the liveloss object data
        
        self.best_model = None
        self.early_stop = early_stop
        if self.early_stop: self.early =  early_stopping(patience=patience, rel_tol=tol)
  
    def cross_validation(self):
        #shuffler = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42).split(train_data, train_labels)
        X = self.X/255.
        y = self.y
        X_train_set = []
        y_train_set = []
        X_val_set = []
        y_val_set = []
        result = []
        
        
        # Store the initial state
        init_state = copy.deepcopy(self.model_ori.state_dict())
        init_state_opt = copy.deepcopy(self.optimiser_ori.state_dict())
        
        # K-fold split
        skf = StratifiedKFold(n_splits= self.n_folds, random_state= self.seed, shuffle=False)
        skf.get_n_splits(X, y)
        for train_index, test_index in skf.split(X, y):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train_set.append(X[train_index]) 
            X_val_set.append(X[test_index])
            y_train_set.append(y[train_index])
            y_val_set.append(y[test_index])
        
        
        for i, data in enumerate(X_train_set):
            self.mean = data.numpy().mean(axis=(0,1,2))#/255
            self.std = data.numpy().std(axis=(0,1,2))#/255
            print(self.mean," " ,self.std)
            
            # create train_transform
            if self.transform:
                train_transform =  transforms.Compose([
                                  transforms.ToPILImage(),
                                  transforms.RandomRotation(10),
                                  transforms.RandomCrop(28, pad_if_needed=True),
                                  transforms.ToTensor(), 
                                  transforms.Normalize(mean=[self.mean], std=[self.std])
                                  ])
            else:
                train_transform =  transforms.Compose([
                                  transforms.Normalize(mean=[self.mean], std=[self.std])
                                  ])
            val_transform = transforms.Compose([
                                transforms.Normalize(mean=[self.mean], std=[self.std])
                                ])
            
            
            #X_train_set[i] = self._Normlization(X_train_set[i].float())
            #X_val_set[i] = self._Normlization(X_val_set[i].float())
            # Check 0-mean, 1-std
            #print("The ", i," train set, mean after normlization: ", X_train_set[i].numpy().mean(axis=(0,1,2)))
            #print("The ", i," train set, std after normlization: ", X_train_set[i].numpy().std(axis=(0,1,2)))
        
        #for i in range(self.n_folds):
            #train_set = TensorDataset(X_train_set[i], y_train_set[i].long())
            #validation_set = TensorDataset(X_val_set[i], y_val_set[i].long())
            
            train_set = CustomTensorDataset(X_train_set[i], y_train_set[i].long(), transform=train_transform)
            validation_set = CustomTensorDataset(X_val_set[i], y_val_set[i].long(), transform=val_transform)
            
            # Reset the model 
            self.model.load_state_dict(init_state)
            self.optimiser.load_state_dict(init_state_opt)
            
            result.append(self._train_validation(train_set, validation_set))
            print("The ", i, " fold finished.")
        ave_result = np.array(result).mean(axis=0)
        
        print(" ")
        print("Result:")
        print("----------------------------")
        print("Average validation loss for ", n ," folds is ", ave_result[0])
        print("Average validation accuracy for ", n ," folds is ", ave_result[1])
        print("Average train loss for ", n ," folds is ", ave_result[2])
        print("Average train accuracy for ", n ," folds is ", ave_result[3])
        return result
      
    def _train_validation(self, train_set, validation_set, plot=True):
        """The train function which takes the weight-decay as the argument """
        t = time.time()

        train_loader = DataLoader(train_set, batch_size= self.batch_size, shuffle=True, num_workers=4)
        validation_loader = DataLoader(validation_set, batch_size= self.test_batch_size, shuffle=False, num_workers=4)
        #test_loader = DataLoader(cifar_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
        if plot:
            liveloss = PlotLosses()
        
        best_val_acc = 0 # Initialize the best score
        
        for epoch in range(self.n_epochs):
            logs = {}
            train_loss, train_accuracy = super().train(train_loader)
            logs['' + 'log loss'] = train_loss.item()
            logs['' + 'accuracy'] = train_accuracy.item()
            validation_loss, validation_accuracy = super().validate(validation_loader)
            logs['val_' + 'log loss'] = validation_loss.item()
            logs['val_' + 'accuracy'] = validation_accuracy.item()
            if plot:
                liveloss.update(logs)
                liveloss.draw()
            logs['time'] = time.time() - t
            
            # Checking stopping criteria
            if self.early_stop: self.early(validation_accuracy)
                
            # Saving the best weights    
            if validation_accuracy.item() > best_val_acc:  
              self.best_model = self.model
              best_val_acc = validation_accuracy.item()
                
            # If the stopping criteria is met  
            if self.early_stop:
                if self.early.stop: 
                    #self.model = self.best_model
                    self.logs = logs
                    print("The best model is found.")
                    break
            self.logs = logs
        self.result = [validation_loss.item(), validation_accuracy.item(), train_loss.item(), train_accuracy.item()]
        return self.result

	  
class early_stopping:
  """
  Counter to implement early stopping
  If validation accuracy has not relative improved above
  a absolute tolerance set by the user than it breaks the 
  training
  If rel_tol is set to 0 it becomes a common counter
  """
  def __init__(self, patience, rel_tol, verbose=True):

    
    self.patience = patience
    self.rel_tol = rel_tol
    self.verbose = verbose
    self.best_score = 0
    self.counter = 0
    self.stop = False

  
  def __call__(self, score):
    
    # If the score is under the required relative tolerance
    # increase the counter is incremented
    if score < self.best_score * (1 + self.rel_tol):
        self.counter += 1
    else:
        self.counter = 0
        
        
    if score > self.best_score:
      self.best_score = score

      
    if self.counter >= self.patience:
      self.stop = True    
      
    if self.verbose:
      print("Count:", self.counter)
	
	
	
def evaluate_test(X_test, model, norm_mean, norm_std, test_batch_size=30, test_transform=None, device="cpu", save_to_csv=False, path="./foo.csv"):
      """
      This method takes a tensor of images and a trained model and returns the predicted labels
      from those images
      Params
      ------
        X_test: torch.tensor of size (no_images, 28, 28), test images dataset
        model: nn.Module or inherited class object
        test_batch_size: int, defines the size of the batch for the datset
        test_transform: transforms.Compose list of transforms to apply to the dataset
        device: str, on which device to run the model
        save_to_csv: bool, option to save predictions to csv in format (index, prediction)
        path: str, path to save string 
        
      Returns
      -------
        y_preds: np.array of predictios made on X_test by the trained model
        
      """
      model.eval()
      model.to(device)
      y_test = torch.zeros_like(X_test)
      test_dataset = CustomTensorDataset(normalise_image(X_test/255., norm_mean, norm_std), y_test, transform=test_transform)
      test_data_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)
      
      y_preds = []
      for X, y in test_data_loader:
          with torch.no_grad():
              X, y = X.to(device), y.to(device)
              a2 = model(X)
              y_pred = F.log_softmax(a2, dim=1).max(1)[1]
              y_preds.append(y_pred.cpu().numpy())
            
      y_preds =  np.concatenate(y_preds, 0)
      
      sub = pd.DataFrame(data={'Category': y_preds})
      sub.index.name = "Id" 
      if save_to_csv:
        sub.to_csv(path)
        
      return y_preds, sub

	  
def model_save(model, name, path, val_acc):
  """Saving function to keep track of models"""
  val = str(val_acc)[2:5]
  path = path + name + '_' + val + '.pth'
  print("Saving model under:", path)
  torch.save(model, path)
  return



def model_load(path, model_name):
  """Loading function for models from google drive"""
  model = torch.load(path + model_name + '.pth')
  return model



def param_strip(param):
  """Strips the key text info out of certain parameters"""
  return str(param)[:str(param).find('(')]



def full_save(path, name, model, optimiser, loss_function, early_stop_tol, n_epoch, lr, momentum, weight_decay, n_folds, train_trans, val_acc, val_loss, train_time, test_acc=None):
  """Saves the models weights and hyperparameters to a pth file and csv file"""
  if train_trans: train_trans="True"
  else: train_trans="False"
  ind = ["Model, Optimiser, Loss Function, Early Stop Tol, Epochs, Learning Rate, Momentum, Weight Decay, nFolds, Augmentations, Val Acc, Val Loss, Training Time, Test Acc"]
  row = [param_strip(model), param_strip(optimiser), param_strip(loss_function), early_stop_tol, n_epoch, lr, momentum, weight_decay, n_folds, train_trans,val_acc, val_loss, train_time, test_acc]
  s = [str(i) for i in row] 
  row = [",".join(s)]
  model_save(model, name, path, val_acc)
  np.savetxt(path + name + '_' + str(val_acc)[2:5] + ".csv", np.r_[ind, row], fmt='%s', delimiter=',')
  return



class ensemble_net(nn.Module):
    """A classifier class that takes individiual pretrained models 
    and aggregates their output values to create ensemble voting
      Params
    ------
      models: a list of model objects
    Returns
    -------
      x_out: a probability vector for the output classes
    """
    def __init__(self, models):
        super(ensemble_net, self).__init__()
        self.models = models
        self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        num = len(self.models)
        for ind, model in enumerate(self.models):
          if ind ==0:
            x1 = model(x)
            x_out = self.soft(x1)
          else:
            x1 = model(x)
            x_out += self.soft(x1)
        x_out /= num
        return x_out
    def inspect(self):
      """Returns the composition of the ensemble model"""
      for model in self.models:
        print(model)
      return None