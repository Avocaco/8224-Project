import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

dim = 2
n = 50
size_lim = 1000

sample = np.random.randint(size_lim, size=(n, dim))

x = np.zeros((n, n))
c = np.repeat(sample[:,None, :], n, axis=1)
c = np.linalg.norm(c-sample, axis=2)
u = np.zeros(n)

plt.scatter(sample[:,0], sample[:, 1])
plt.show()

def plot(order, sample, n):
    ordered_sample = sample[order+[order[0]]]
    plt.scatter(sample[:,0], sample[:, 1], c='r')
#     for i in range(n):
#         plt.annotate(str(i), xy = (sample[i,0], sample[i,1]))
        
    plt.plot(ordered_sample[:, 0], ordered_sample[:, 1])
    
    plt.show()
    
    
    def objective(c, x):
    return np.sum(x*c)

def get_order(x, n):
    order = [0]
    
    for _ in range(n-1):
        order.append(np.argmax(x[order[-1], :]))
        
    return order

def x_check(x):
    if np.any(np.sum(x, axis=0) != 1):
        return False
    if np.any(np.sum(x, axis=1) != 1):
        return False
    
    return True

def greedy_initialize(c, n):
    x = np.zeros((n, n))
    c = c+(np.max(c)+1)*np.eye(n)
    i = 0
    for _ in range(n-1):
        j = np.argmin(c[i,:])
        x[i, j] = 1
        c[i, :] = np.Inf
        c[:, i] = np.Inf
        i = j
    
    x[j, 0] = 1
    order = get_order(x, n)
    
    return x, order

x = greedy_initialize(c, n)


def order_swap(x, order,  n):
    i = random.randint(0, n-1)
    pre, succ = np.argmax(x[:, i]), np.argmax(x[i, :])
    succ_succ = np.argmax(x[succ, :])
    x[pre, i], x[i, succ], x[succ, succ_succ] = 0, 0, 0
    x[pre, succ], x[succ, i], x[i, succ_succ] = 1, 1, 1
    
    order[order==i], order[order==succ] = succ, i
    
    return x, order
    
def succ_swap(x, order, n):
    o1, o2 = np.random.choice(range(n), 2)
    
    pre1,i1, succ1 = order[o1-1], order[o1], order[(o1+1)%n]
    pre2,i2, succ2 = order[o2-1], order[o2], order[(o2+1)%n]
    
    x[pre1, i1], x[i1, succ1] = 0, 0
    x[pre2, i2], x[i2, succ2] = 0, 0
    
    order[o1], order[o2] = order[o2], order[o1]
    
    x[order[o1-1], order[o1]] = 1
    x[order[o1], order[(o1+1)%n]] = 1
    x[order[o2-1], order[o2]] = 1
    x[order[o2], order[(o2+1)%n]] = 1
    
    return x, order
    
    
    def step(x, order, n):
    choice = random.randint(1, 1)

    if choice == 0:
        return order_swap(x, order, n)
    elif choice == 1:
        return succ_swap(x, order, n)
        
        
def simulated_annealing_TSP(c, n, max_attempts = 200, T=200, T_min=1, decay = 0.99, max_iter = 200):
    x, order = greedy_initialize(c, n)
    current_obj = objective(c, x)
    
    best_x, best_order, best_obj = x, order, current_obj

    print(f'Initial objective: {current_obj}')
    s = time.time()
    
    for epoch in range(max_iter):
        if T < T_min:
            break
            
        for _ in range(max_attempts):
            xc, orderc = x.copy(), order.copy()
                
#                 if not x_check(xc):
#                     print('Bad step method')
#                     return
            
            xc, orderc = step(xc, orderc, n)
            obj = objective(c, xc)
            if obj <= current_obj or random.random()< math.exp((current_obj-obj)/T):
                current_obj = obj
                x = xc
                order = orderc
            
            if obj < best_obj:
                best_obj = obj
                best_x = xc
                best_order = orderc
                
        T *= decay
        
        if not (epoch+1)%100:
            print(f'Epoch: {epoch+1}, Best obj: {best_obj:.2f}, Current obj: {current_obj:.4f} Time: {time.time()-s:.4f} s')
            s = time.time()
        
    return best_x, best_order
    

max_attempts = 100
T = 200
T_min = 0.01
decay = 0.995
max_iter = 10000

x, order = simulated_annealing_TSP(c, n, max_attempts, T, T_min, decay, max_iter)

plot(order, sample, n)
