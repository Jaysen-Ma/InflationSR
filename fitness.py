import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar
import seaborn as sns
import sympy as sy
import warnings
warnings.filterwarnings("ignore")

def setup(V,k):
  
  # Filling up the functions to numpy ndarray to determine the minimum value of the potential using the scipy package
  x = sy.Symbol('x')
  y = sy.lambdify(x, V, 'numpy')
  min_x = minimize_scalar(y, bounds=(-5, 5), method='bounded')

  # redefine the potential such that V >= 0 within the range probed and V(phi_min) = 0
  #V = V.subs(x,min_x.x+x)
  if min_x.fun < 0:
    V += abs(min_x.fun) 
  if min_x.fun > 0:
    V -= min_x.fun
  
  y = sy.lambdify(x, V, 'numpy')
  y_prime = sy.lambdify(x, V.diff(x), 'numpy')
  y_pprime = sy.lambdify(x, V.diff(x).diff(x), 'numpy')
  epsilon = sy.lambdify(x, (1/2)*(V.diff(x)/V)**(2), 'numpy')
  eta = sy.lambdify(x, V.diff(x).diff(x)/V, 'numpy')
  n_int = sy.lambdify(x, (V/V.diff(x)), 'numpy')
  eqn_4 = sy.lambdify(x,(np.pi*(np.abs(V.diff(x))/(3*V))-((V/3)**(1/2))/(2*np.pi)),'numpy')

  return x, y, y_prime, y_pprime, epsilon, eta, n_int, eqn_4, min_x.x

def fit_fun(V): # this functions calculate the fitness value of any given potential V
  print('Current function: ',V)
  k = 20
  #if V is a constant function, the setup function will not work, reutrn fitness -100 for such potentials
  try: 
    x_ = np.linspace(-k,k,10000)
    x, y, y_prime, y_pprime, epsilon, eta, n_int, eqn_4, Phi_min = setup(V,k)
    for array in [y(x_), y_prime(x_), y_pprime(x_), epsilon(x_), eta(x_), n_int(x_)]:
      if np.any(np.isnan(array)):
        return -30 # there might be a chance the cosmological observables have values 'nan', in such case return -30
  except:
    return -100

  delta = 2*k/10000 # width of each step between the interval [-k,k]
  Phi_l = Phi_min #start exploring to the left hand side of the 'global minimum'
  Phi_r = Phi_min #dummy variable used to explore the right hand side
    
  try:
    #finding phi such that II.3 are satisfied
    # 1. Iterating to the right of the minimum:
    while Phi_r < k:
        if np.abs(eta(Phi_r)) > 0.5 or epsilon(Phi_r) > 0.5:
          Phi_r += delta
        else:
          Phi_er = Phi_r # Right end point
          while Phi_r < k:
            if np.abs(eta(Phi_r)) < 0.5 and epsilon(Phi_r) < 0.5 and eqn_4(Phi_r) > 0 and y(Phi_r+delta) > y(Phi_r):
              Phi_r += delta  
            else:
              Phi_ir = Phi_r #Right start point
              break
          break
      # 2. Iterating to the left of the minimum:
    while Phi_l > -k:
        if np.abs(eta(Phi_l)) > 0.5 or epsilon(Phi_l) > 0.5:
          Phi_l -= delta
        else:
          Phi_el = Phi_l # Left end point
          while Phi_l > -k:
            if np.abs(eta(Phi_l)) < 0.5 and epsilon(Phi_l) < 0.5 and eqn_4(Phi_l) > 0 and y(Phi_l-delta) > y(Phi_l):
              Phi_l -= delta  
            else:
              Phi_il = Phi_l #Left start point
              break
          break
 
      # if local minimum and ending point found, but no startinging point, set starting point to be at the boundary, 
      # if local minimum found but not ending point of inflation on both sides, assign fitness = -25
    f_left = 0
    f_right = 0
  
    if Phi_r > k:
        try:
          Phi_er = Phi_er
          Phi_ir = k
        except: # no ending point of inflation found on right hand side
          Phi_er = k
          Phi_ir = k
          f_right = 1
    if Phi_l < -k:
        try:
          Phi_el = Phi_el
          Phi_il = -k
        except:
          Phi_el = -k
          Phi_il = -k
          f_left = 1
    if f_left == 1 and f_right == 1:
        return -25
  
    f_term = -0.3

    def N_total(Phi_e,Phi_i):
        sum = np.abs(integrate.quad(n_int, Phi_i, Phi_e)[0])
        return sum

    def N_star(Phi_e):
        N_star = 58 + np.log(y(Phi_e))/6
        return N_star

    def phi_star(Phi_e, Phi_min):
        n_star = N_star(Phi_e)
        sum = 0
        Phi = Phi_e
        if Phi_min > Phi:
          while sum < n_star and Phi > -k:
            sum = np.abs(integrate.quad(n_int, Phi_e, Phi)[0])
            Phi -= delta
          return Phi
        else:
          while sum < n_star and Phi < k:
            sum = integrate.quad(n_int, Phi_e, Phi)[0]
            Phi += delta
          return Phi

    def n_star(phi_star):
        ns = 1 - 6*epsilon(phi_star) + 2*eta(phi_star)
        return ns

    def r_star(phi_star):
        r = 16*epsilon(phi_star)
        return r
    
    def m_star(phi_star):
        ms = (y(phi_star)/epsilon(phi_star))**(1/4)
        return ms

    def fitness(Phi_e,Phi_i,phi_star):
        phi_star = phi_star    
        N_star_value = N_star(Phi_e)
        N_total_value = N_total(Phi_e, Phi_i)
        if N_star_value > N_total_value:
          f1 = -(N_star_value - N_total_value)/10
        else:
          f1 = 0
  
        nstar = n_star(phi_star)
        rstar = r_star(phi_star)
        mstar = m_star(phi_star)
    
        f_ns = -np.log10(1+np.abs(nstar-0.9649)/0.0042)
        if f_term > f_ns:
          f2 = f_ns
        else:
          f2 = 0
  
        if rstar > 0.061:
          f3 = -np.log10(r_star(phi_star)/0.061)
        else:
          f3 = 0

        f_m = -np.log10(1+np.abs(mstar-0.027)/0.0027)
        if f_term > f_m:
          f4 = f_m
        else:
          f4 = 0

        fit = f1 + f2 + f3 + f4
        return fit, N_star_value, N_total_value, nstar, rstar, mstar, phi_star

    fit = []
    if f_left != 1:
        Phi_star_l = phi_star(Phi_el, Phi_min)
        left_fit, N_star_l, N_total_l, n_star_l, r_star_l, m_star_l, phi_star_l = fitness(Phi_el,Phi_il,Phi_star_l)
        fit.append(left_fit)
        if left_fit == 0:
            printing(Phi_l,left_fit,Phi_min,Phi_il,Phi_el,N_total_l,N_star_l,phi_star_l,n_star_l,r_star_l,m_star_l,y,k)
    else:
        Phi_star_l = -k
        left_fit = -15
        fit.append(left_fit)
    
    if f_right != 1:
        Phi_star_r = phi_star(Phi_er, Phi_min)
        right_fit, N_star_r, N_total_r, n_star_r, r_star_r, m_star_r, phi_star_r = fitness(Phi_er,Phi_ir,Phi_star_r)
        fit.append(right_fit)
        if right_fit == 0:
            printing(Phi_r,fit,Phi_min,Phi_ir,Phi_er,N_total_r,N_star_r,phi_star_r,n_star_r,r_star_r,m_star_r,y,k)
    else:
        Phi_star_r = k
        right_fit = -15
        fit.append(right_fit)
  except OverflowError:
        return -50
  print(fit)   
  return max(fit)

def printing(Phi,fit,Phi_min,Phi_i,Phi_e,N_total,N_star,phi_star,n_star,r_star,m_star,y,k):
    x_ = np.linspace(-k,k,10000)
    x_1=np.linspace(Phi_e,Phi_i,5000)
   
    plt.plot(x_, y(x_), label='label')
    plt.plot(x_1,y(x_1),label='rolling region')
    
    plt.title("Potential plot")
    plt.xlabel("Φ")
    plt.ylabel("V(Φ)")
    plt.legend()
    plt.show()
    
    print('fitness = ', fit)
    print('Phi at the end of iteration =', Phi)
    print('Phi_min =', Phi_min)
    print('(Start point, End point) =', (Phi_i,Phi_e))
    print('efold =', N_total)
    print('N_star =', N_star)
    print('phi_star =', phi_star)
    print('n_star =', n_star)
    print('r_star =', r_star)
    print('m_star =', m_star)