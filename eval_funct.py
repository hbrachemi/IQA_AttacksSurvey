import numpy as np
import scipy
from scipy.optimize import curve_fit
from tqdm import tqdm
from config import *
from sklearn.metrics import mean_squared_error


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE]
  
def evaluate(val_dl, metric, transf, noise_list ,image_distance_metric):
  mos = list()
  y_adv = list()
  fr = list()
  
  for i ,[inputs, labels] in enumerate(tqdm(val_dl)):
                inputs = inputs.to(device)
<<<<<<< HEAD
=======
                init_inputs = torch.clone(inputs)
>>>>>>> ebdce8ef1760308e9bdb928ac6c8b25b8f2f9a93
                mos.append(float(labels.cpu().detach()))
                if noise_list is None:
                   noise = noise_list[i]
                   inputs = inputs + noise
                if transf is not None:
                   inputs = transf(inputs)
                y_adv.append(float(metric(inputs).cpu().detach()))
                if image_distance_metric is not None:
<<<<<<< HEAD
                   fr.append(image_distance_metric(inputs,inputs))
=======
                   fr.append(image_distance_metric(inputs,init_inputs))
>>>>>>> ebdce8ef1760308e9bdb928ac6c8b25b8f2f9a93
  [SRCC, KRCC, PLCC, RMSE]=compute_metrics(mos,y_adv)
  
  return SRCC,KRCC,PLCC,RMSE,fr         
  
