# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 12:45:38 2013

@author: joses

@description: Product-Difference algorithm.
Gordon, R. G. (1968). Error bounds in equilibrium statistical mechanics.
Journal of Mathematical Physics, 9(5):655-663.
"""

import numpy as np
import scipy as sp

def PD_build_Pij(mom):
  """
   Build Pij matrix, part of PD algorithm.
  """
  N   = int(len(mom)/2) #-- number of quad point
  m   = 2*N             #-- number of moments
  pIJ = sp.zeros((m+1, m+1))

  "-- 1st column; j=0"
  for j in [0]:
    for i in range(m+1):
      pIJ[i][0] = 0; pIJ[0][0] = 1
  "-- 2nd column; j=1"
  for j in [1]:
    for i in range(m):
      pIJ[i][1] = (-1)**i * mom[i]
  "-- 3rd column; j>=2"
  for j in range(2,m+1):
    for i in range(m+2-j):
      pIJ[i][j] = pIJ[0][j-1]*pIJ[i+1][j-2] - pIJ[0][j-2]*pIJ[i+1][j-1]

  return pIJ

def PD_build_alphaJ(Pij):
  """
   Build alpha vector, part of PD algorithm.
  """
  m = int(len(Pij[0])-1)
  "-- Trought 1st line; i=0"
  alphaJ = sp.zeros(m)
  for i in range(1,m):
    if Pij[0][i]*Pij[0][i-1]>0:
      alphaJ[i] = Pij[0][i+1]/(Pij[0][i]*Pij[0][i-1])
    else:
      #print "Warning: value set to zero!"
      alphaJ[i] = 0.

  return alphaJ

def PD_build_Jn(alphaJ):
  """
   Build Jn tridiagonal-matrix, part of PD algorithm.
  """
  N = int(len(alphaJ)/2)
  "1. Main-diagonal Adiag of TDM"
  Adiag = sp.zeros(N)
  for i in range(N):
    Adiag[i] = alphaJ[2*i] + alphaJ[2*i+1]
  "2. Off-diagonals Asup and Asub"
  Asub = sp.zeros(N-1);  Asup = sp.zeros(N-1)
  for i in range(N-1):
    Asub[i] = -np.sqrt(alphaJ[2*i+2]*alphaJ[2*i+1])
    Asup[i] = -np.sqrt(alphaJ[2*i+2]*alphaJ[2*i+1])
  "3. Assembling the tri-diagonal matrix: Jn"
  Jn = np.diag(Asub, -1) + np.diag(Adiag, 0) + np.diag(Asup, 1)
  
  return Jn

def quadrature_from_ab(a, b):
    sec_diag = -np.sqrt(np.abs(b[1:]))
    Jn = np.diag(sec_diag, -1) + np.diag(a, 0) + np.diag(sec_diag, 1)
    "-- calculate the eigenvalues and eigenvectors of Jn"
    e_vals, e_vecs = np.linalg.eig(Jn)
    p_sort = e_vals.argsort()
    e_vals = e_vals[p_sort]
    e_vecs = e_vecs[0, :]
    e_vecs = e_vecs[p_sort]
    "-- get the results: abscissas and weigths"
    xi = e_vals
    w  = e_vecs**2
    return (xi, w)

def ProductDifference(mom):
  """
   Product-Difference algorithm. Main sub-routine.
  """
  #np.set_printoptions(precision=2)
#  verb_a = False #-- verbosity for debug

  N = int(len(mom)/2)
  "1. build matrix pIJ"
  pIJ = PD_build_Pij(mom)
  "2. build vector alphaJ"
  alphaJ = PD_build_alphaJ(pIJ)
  "3. build Jn - Tri-diagonal Matrix"
  Jn = PD_build_Jn(alphaJ)
  "4. calculate the eigenvalues and eigenvectors of Jn"
  e_vals, e_vecs = np.linalg.eig(Jn)
  p_sort = e_vals.argsort()
  e_vals.sort(); e_vecs = e_vecs[p_sort]
  "5. get the results: abscissas and weigths"
  " 5.1 - the abscissas are equal to the eigenvalues and the"
  " 5.2 - the weights are equal to the square of the first elements of the eigenvectors"
  xi = e_vals
  w  = np.zeros(N)
  for i in range(N):
    w[i] = e_vecs[0][i]**2 * mom[0]

#  if verb_a:
    #print "<<MoM>> \n", mom
    #print "\n <<Matrix Pij>> \n", pIJ,"\n"
#    print("<<Vector>> \n",alphaJ,"\n")
    #print "\n <<TDM>> \n", Jn
    #print "\n ::Eigen-Val:: \n", e_vals
    #print   " ::Eigen-Vec:: \n", e_vecs
    #print "\n"
    #for i in range(N):
    #  print "Abscissas",i,"=",xi[i]
    #print "\n"
    #for i in range(N):
    #  print "Wight",i,"=", w[i]

  return xi,w
  
def Wheeler(mom):
  """
   Compute quadrature approximation with the Wheeler algorithm.
   
   Sack and Donovan (1971). An algorithm for Gaussian quadrature given modified moments.
   Numerische Mathematik, 18(5):465-478. 
   
   Wheeler (1974). Modified moments and Gaussian quadratures. 
   Rocky Mountain Journal of Mathematics 4, 287–296.
  """
  N = int(len(mom)/2)
  m = 2*N
  "-- compute intermediate coefficients "
  sigma = sp.zeros((N+1, m))
  sigma[1,:] = mom[:]
  "-- compute coefficients for Jn matrix"
  a = sp.zeros(N)
  b = sp.zeros(N)
  a[0] = mom[1]/mom[0]
  for k in range(N-1):
    for l in range(k,m-k-2):
      sigma[k+2][l+1] =  sigma[k+1][l+2] - a[k]*sigma[k+1][l+1] - b[k]*sigma[k][l+1]
      a[k+1]          = -sigma[k+1][k+1]/sigma[k+1][k] + sigma[k+2][k+2]/sigma[k+2][k+1]
      b[k+1]          =  sigma[k+2][k+1]/sigma[k+1][k]
  xi, w = quadrature_from_ab(a, b)
  w *= mom[0]
    
  return xi, w

def ZetaChebyshev(mom_in, wratio_min=0.0):
    """

    """
    mom = mom_in / mom_in[0]
    cutoff = 0.0
    N = int(len(mom)/2)
    m = 2*N
    "-- compute intermediate coefficients "
    sigma = sp.zeros((N+1, m))
    sigma[1,:] = mom[:]
    "-- compute coefficients for Jn matrix"
    a = sp.zeros(N)
    b = sp.zeros(N)
    zeta = sp.zeros(2*N-1)
    a[0] = mom[1]/mom[0]
    zeta[0] = a[0]
    for k in range(N-1):
        for l in range(k,m-k-2):
            sigma[k+2][l+1] =  (sigma[k+1, l+2] - a[k]*sigma[k+1, l+1] - 
                               b[k]*sigma[k, l+1])
            a[k+1]          = (-sigma[k+1, k+1]/sigma[k+1, k] + 
                              sigma[k+2, k+2]/sigma[k+2, k+1])
            b[k+1]          =  sigma[k+2, k+1]/sigma[k+1, k]
        zeta[2*k+1] = b[k+1] / zeta[2*k]
        zeta[2*k+2] = a[k+1] - zeta[2*k+1]

    zeta = np.nan_to_num(zeta)
    zerozeta = zeta <= cutoff
    if zerozeta.any():
        nkmax = np.argmax(zerozeta)
        assert nkmax > 0
        N = np.floor_divide(nkmax - 1, 2) + 1
    
    for n in range(N, 0, -1):
        xi, w = quadrature_from_ab(a[:n], b[:n])
        w *= mom_in[0]
        if np.amin(w) / np.amax(w) >= wratio_min:
            break
        
    return xi, w