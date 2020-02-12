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
    p = 1/(mu.max()-mu.min())
    return p

def loglike(y, sigma, mu):
    """
    Likelihod de tener un dato x e incertidumbre sigma
    """
    L=0
    for i in range(len(Y)):
        L +=  np.log((1.0/np.sqrt(2.0*np.pi*sigma[i]**2)))-(0.5*(y[i]-mu[i])**2/(sigma[i]**2))
    return L

def modelo(X,beta,beta0):
    
    mu=X.dot(beta)+beta0*np.ones(len(X))
    return mu 

def posterior(mu, y, sigma):
    """
    Posterior calculado con la normalizacion adecuada
    """
    post =  like(y, sigma, mu) * prior(mu)
    return  post

#Xnew.shape

#inicializamos los betas
dimbetas=X.shape[1]
betas=np.ones(dimbetas)
beta0=1
sigmas=0.1*np.ones(len(Y))


#si queremos más resolucion en los pasos
paso=1

#el número de pasos que voy a realizar
n=10

#diccionario de betas
todosbetas=[]
todosbeta0=[]
todosbetas.append(betas)
todosbeta0.append(beta0)
#el algoritmo
for i in range(n):
    ranStep0=paso*(np.random.uniform(-1,1))
    beta0new=beta0+ranStep0
    betasnew=np.zeros(len(betas))
    
    #Hace el paso de cada beta aleatorio
    for i in range(len(betas)):
        betasnew[i]=betas[i]+ paso*(np.random.uniform(-1,1))
        
    munew=modelo(X,betasnew,beta0new)
    muantes=modelo(X,betas,beta0)
    
    probnew=posterior(munew,Y,sigmas)
    probantes=posterior(muantes,Y,sigmas)
    
    if probnew>=probantes:
        betas=betasnew
        beta0=beta0new
    else:
        alpha=np.random.rand()
        if alpha<probnew/probantes:
            betas=betasnew
            beta0=beta0new
            
    todosbetas.append(betas)
    todosbeta0.append(beta0)
    
    
print(todosbetas)
print(todosbeta0)
            
