import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model


# lee los datos
data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]

def prior(mu):
    """
    Densidad de probabilidad de mu
    """
    p = np.ones(len(mu))/(mu.max()-mu.min())
    return p

def like(y, sigma, mu):
    """
    Likelihod de tener un dato x e incertidumbre sigma
    """
    L = np.ones(len(mu))
    for y_i,sigma_i in zip(y, sigma):
        L *= (1.0/np.sqrt(2.0*np.pi*sigma_i**2))*np.exp(-0.5*(y_i-mu)**2/(sigma_i**2))
    return L

def modelo(X,beta):
    mu=X.dot(beta)
    return mu 

def posterior(mu, y, sigma):
    """
    Posterior calculado con la normalizacion adecuada
    """
    post =  like(y, sigma, mu) * prior(mu)
    evidencia = np.trapz(post, mu)
    return  post/evidencia





        