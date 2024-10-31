import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from point.helper import get_rff_model, method
from point.misc import Space
from point.laplace import opt_method
from point.optim.optim_function import get_optim_func_deep
from point.utils import build_coal_data, domain_grid_1D, normalize , domain_grid_2D

# 1D data

## synthetic data 1
test_X = np.linspace(0, 10, 1000).reshape(-1, 1)
event_time = np.load('events.npy')
intensity = np.load('intensity.npy')

plt.plot(event_time, np.zeros_like(event_time), "k|", label = 'Events')
plt.plot(np.squeeze(test_X), intensity, color='g', label = 'Ground Truth')
# plt.plot(np.squeeze(test_X), sample**2, color='g', label = 'truth')

plt.xlabel("$x$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper right', prop={'size': 8})
plt.savefig('syn1.pdf', bbox_inches = 'tight')
plt.show()

rng = np.random.RandomState(120)

scale = 1
domain = [0,10]

X =  normalize(event_time, domain, scale).reshape(-1,1)
space =  Space(scale = scale) 
grid =  domain_grid_1D(space.bound1D, 200)
ogrid =  domain_grid_1D(domain, 200)
shift = (domain[1]-domain[0])/space.measure(1)

md = get_rff_model(name = 'deep', n_components = [100, 50], sample = 1000, m = [0,0], d = [0.8,0.9], n_dims = 1,  space = space, random_state = rng)
md.lrgp.beta0.assign( md.lrgp.beta0 ) 
ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
ofunc(md, X, verbose= True)
ld = md.predict_lambda(grid) / shift
l2 = sum((intensity-ld)**2).numpy()
l2 = np.sqrt(l2[0]) / 1000

################## print
_, lower, upper = md.predict_lambda_and_percentiles(grid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(7, 4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(ogrid.min(), ogrid.max())

plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'Predict 10-90%' )

# plt.plot(ogrid, lv, 'b:', label='VBPP')
# plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
# plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, ld, 'r:', linewidth=1.2, label='Predict')
# plt.plot(ogrid, lr1, 'b:', linewidth=1.2, label='SSPP')
# plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

oX_train =  X * (domain[1] - np.mean(domain)) / scale + np.mean(domain)
plt.plot(oX_train, np.zeros_like(oX_train), "k|", label = 'Events')
plt.plot(np.squeeze(test_X), intensity, color='g', label = 'Ground Truth')
# plt.plot(np.squeeze(test_X), sample**2, color='g', label = 'truth')

plt.xlabel("$x$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper right', prop={'size': 8}, bbox_to_anchor=(1.0, 1.0))
plt.show()

## synthetic data 2

test_X = np.linspace(0, 10, 1000).reshape(-1, 1)
event_time = np.load('events_nos.npy')
intensity = np.load('intensity_nos.npy')

plt.plot(event_time, np.zeros_like(event_time), "k|", label = 'Events')
plt.plot(np.squeeze(test_X), intensity, color='g', label = 'Ground Truth')
# plt.plot(np.squeeze(test_X), sample**2, color='g', label = 'truth')

plt.xlabel("$x$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper left', prop={'size': 8})
plt.savefig('syn2.pdf', bbox_inches = 'tight')
plt.show()

rng = np.random.RandomState(120)

scale = 1
domain = [0,10]

X =  normalize(event_time, domain, scale).reshape(-1,1)
space =  Space(scale = scale) 
grid =  domain_grid_1D(space.bound1D, 200)
ogrid =  domain_grid_1D(domain, 200)
shift = (domain[1]-domain[0])/space.measure(1)

md = get_rff_model(name = 'deep', n_components = [100, 50], sample = 1000, m = [0,0], d = [1,1], n_dims = 1,  space = space, random_state = rng)
md.lrgp.beta0.assign( md.lrgp.beta0 ) 
ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
ofunc(md, X, verbose= True)
ld = md.predict_lambda(grid) / shift

################## print
_, lower, upper = md.predict_lambda_and_percentiles(grid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(7, 4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(ogrid.min(), ogrid.max())

plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'Predict 10-90%' )

# plt.plot(ogrid, lv, 'b:', label='VBPP')
# plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
# plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, ld, 'r:', linewidth=1.2, label='Predict')
# plt.plot(ogrid, lr1, 'b:', linewidth=1.2, label='SSPP')
# plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

oX_train =  X * (domain[1] - np.mean(domain)) / scale + np.mean(domain)
plt.plot(oX_train, np.zeros_like(oX_train), "k|", label = 'Events')
plt.plot(np.squeeze(test_X), intensity, color='g', label = 'Ground Truth')
# plt.plot(np.squeeze(test_X), sample**2, color='g', label = 'truth')

plt.xlabel("$x$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper right', prop={'size': 8}, bbox_to_anchor=(1.0, 1.0))
plt.show()

## Coal data

def normalize(X, domain, scale = 1.0) :
    center = np.mean(domain)
    norm = domain[1] - center
    return scale * (X - center) / norm

rng = np.random.RandomState(150)

scale = 2
X, domain = build_coal_data()
X = normalize(X, domain, scale)
space =  Space(scale = scale) 
grid =  domain_grid_1D(space.bound1D, 100)
ogrid =  domain_grid_1D([1850, 1962], 100)
shift = 11500

md = get_rff_model(name = 'deep', n_components = [50, 30], sample = 1000, m = [0,0], d = [0.8,0.8], n_dims = 1,  space = space, random_state = rng)
md.lrgp.beta0.assign( md.lrgp.beta0 ) 
ofunc = get_optim_func_deep(n_loop = 2, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
ofunc(md, X, verbose= True)
ld = md.predict_lambda(grid) / shift

################## print
_, lower, upper = md.predict_lambda_and_percentiles(grid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(7, 4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(ogrid.min(), ogrid.max())

plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'Predict 10-90%' )

# plt.plot(ogrid, lv, 'b:', label='VBPP')
# plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
# plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, ld, 'r:', linewidth=1.2, label='Predict')
# plt.plot(ogrid, lr1, 'b:', linewidth=1.2, label='SSPP')
# plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

oX_train =  X * (domain[1] - np.mean(domain)) / scale + np.mean(domain)
plt.plot(oX_train, np.zeros_like(oX_train), "k|", label = 'Events')
# plt.plot(np.squeeze(test_X3), lam3, color='g', label = 'Ground Truth')
# plt.plot(np.squeeze(test_X), sample**2, color='g', label = 'truth')

plt.xlabel("$x$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper right', prop={'size': 8}, bbox_to_anchor=(1.0, 1.0))
plt.show()

# 2D data

def print_grid_2D(grid, lambda_sample, save, X = None, X2 = None, vmin = None, vmax = None, name = None , colorbar = False, figsize = (5,4.5)):
    n = int(np.sqrt(lambda_sample.shape[0]))
    
    if tf.is_tensor(lambda_sample):
         lambda_sample = lambda_sample.numpy()
         
    lambda_matrix = lambda_sample.reshape(n,n)
    cmap = 'YlOrRd'
    #cmap = 'viridis'
    #cmap = 'Reds'
    
    plt.figure(figsize=figsize)
    plt.xlim(grid.min(), grid.max())
    plt.pcolormesh(np.unique(grid[:,0]), np.unique(grid[:,1]), lambda_matrix, vmin = vmin, vmax = vmax, shading='auto', cmap= cmap)
    # CS = plt.contour(grid[:,0], grid[:,1], gaussian_filter(Z, 5.), 4, colors='k',interpolation='none')
    
    if X is not None :
        plt.plot(X[:,0], X[:,1], 'wo', markersize = 2)
        plt.plot(X[:,0], X[:,1], 'k.', markersize = 2)
        
    if X2 is not None :
        plt.plot(X2[:,0], X2[:,1],'rx', markersize = 1)
    
    if name is not None : 
        plt.title(name)

    #plt.xticks(fontsize=8)
    #plt.yticks(fontsize=8)
    plt.axis('off')

    if colorbar is True :
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=8)
    
    if save[0]:
        plt.savefig(save[0], bbox_inches = 'tight')

    plt.show()

## Red Wood data

def get_rw_data_set(scale = 1.0):
    name = 'redwood'
    directory = "./data"
    data = np.genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = scale * data[1:, 0:2]
    return data

rng  = np.random.RandomState(125)
X = get_rw_data_set(scale = 2)
space = Space(scale = 2)

grid, _,_ = domain_grid_2D(bound = space.bound1D , step = space.bound1D[1]/100)
ogrid = (grid + 2) / 4
oX = (X + 2) / 4

variance = 1.0
l = [0.3, 0.3]
tol = 1e-06

md = get_rff_model(name = 'deep',  n_dims = 2, n_components = [50, 30], sample = 1000, m = [0,0], d = [0.8, 0.8], space = space, random_state = rng)
md.lrgp.beta0.assign( md.lrgp.beta0 ) 
md.default_opt_method = opt_method.NEWTON_CG
ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
ofunc(md, X, verbose= True)

lambda_mean_d = md.predict_lambda(grid).numpy()
# lambda_mean = m.predict_lambda(grid).numpy()
print_grid_2D(ogrid, lambda_mean_d, ['rw_md.pdf'], oX, vmin = 0, vmax = 70, colorbar = True, figsize = (4.8,3.6)) 

## Taxi data

def normalize(X, domain, scale = 1.0) :
    center = np.mean(domain)
    norm = domain[1] - center
    return scale * (X - center) / norm

def get_taxi_data_set(scale = 1.0):
    rng  = np.random.RandomState(200)
    directory = "./data"
    name = "porto_trajectories"
    data = np.genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = data[data[:,1] <= 41.18]
    data = data[data[:,1] >= 41.147]
    data = data[data[:,0] <= -8.58]
    data = data[data[:,0] >= -8.65]

    data = data[rng.choice(data.shape[0], 4000, replace=False), :]

    X1 =  normalize(data[:,0],  [-8.58, -8.65], scale = 1.0)
    X2 =  normalize(data[:,1],  [41.147, 41.18], scale = 1.0)
    data = scale * np.column_stack((X1,X2))
    
    return data

rng  = np.random.RandomState(200)
scale = 2
X = get_taxi_data_set(scale = scale)
space = Space(scale = scale)
shift = 15

models = []
variance = 1.0
l = [0.3, 0.3]
tol = 1e-06

grid, _,_ = domain_grid_2D(bound = space.bound1D , step = space.bound1D[1]/100)
ogrid = (grid + 2) / 4
oX = (X + 2) / 4

md = get_rff_model(name = 'deep', n_dims = 2, n_components = [50, 30], sample = 1000, m = [0,0], d = [0.8, 0.8], space = space, random_state = rng)
md.lrgp.beta0.assign( md.lrgp.beta0 ) 
md.default_opt_method = opt_method.NEWTON_CG
# The epoch can be adjusted.
ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 30, wage_decay = True)
ofunc(md, X, verbose= True)

lambda_mean_d = md.predict_lambda(grid).numpy()
# lambda_mean = m.predict_lambda(grid).numpy()
print_grid_2D(ogrid, lambda_mean_d,['taxi_md.pdf'] ,oX, vmin = 0, vmax = 2000, colorbar = True, figsize = (4.8,3.6)) 

