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

def modelo(X,beta,beta0):
    
    mu=X.dot(beta)+beta0*np.ones(len(X))
    return mu 

def posterior(mu, y, sigma):
    """
    Posterior calculado con la normalizacion adecuada
    """
    post =  like(y, sigma, mu) * prior(mu)
    evidencia = np.trapz(post, mu)
    return  post/evidencia

#Xnew.shape

#inicializamos los betas
dimbetas=X.shape[1]
betas=np.ones(dimbetas)
beta0=1
sigmas=0.1*np.ones(len(Y))


#si queremos más resolucion en los pasos
paso=1

#el número de pasos que voy a realizar
n=1000

#diccionario de betas
todosbetas=[]
todosbetas.append(betas)
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
    


        
