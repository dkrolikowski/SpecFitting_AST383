from sklearn.decomposition import PCA
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np

import bspline_acr as bspl

import h5py, scipy.linalg, pdb

### Function to turn the hdf5 grid output from Starfish itself into a numpy array
#   Also returns the wavelength array and an array of temps corresponding to each entry in the template grid

def contfit( flux ):
    
    cont = flux.copy()
    xarr = np.arange( flux.size, dtype = np.float )
    
    spl  = bspl.iterfit( xarr, flux, bkspace = 1250, upper = 2.0, lower = 0.2, maxiter = 15, nord = 3 )[0]
    cont = spl.value( xarr )[0]
    
    return flux / cont

def makeGridNumpy():
    
    gridIN     = h5py.File( 'grid.hdf5', 'r' )
    
    model_wl   = np.array( gridIN['wl'] )
    model_flux = np.zeros( ( gridIN['pars'].shape[0], model_wl.size ) )
    model_Teff = np.transpose( gridIN['pars'] )[0]
    
    for i in range( model_Teff.size ):
        
        modstr        = 't' + str( int( model_Teff[i] ) ) + 'g4.5Z0.0'
        
        thisflux      = np.array( gridIN['flux'][modstr] )
        model_flux[i] = contfit( thisflux )

    return model_flux, model_wl, model_Teff

### Function to whiten the templates
#   Returns the whitened grid, can be made to return the mean and standard deviation as well
    
def whitenTemplates( models, return_musig = False ):
    
    template_mean = np.mean( models, axis = 0 )
    template_std = np.std( models, axis = 0 )
        
    models_whtnd  = ( models - template_mean ) / template_std
    
    if return_musig: return models_whtnd, template_mean, template_std
    else: return models_whtnd

### Function that takes in the model grid fluxes and returns the eigenspectra
#   Also takes in the fraction of the variance you want to allow
#   Also returns the number of PCA components kept for the variance fraction given

def getPCA_Eigenspec( grid, varfrac = 0.98 ):
    
    pcatest     = PCA( n_components = 30 )
    pcatest_fit = pcatest.fit( grid )
    
    compsneed   = np.where( np.cumsum( pcatest_fit.explained_variance_ratio_ ) >= varfrac )[0][0] + 1
    
    pcaout      = PCA( n_components = compsneed )
    pcafit      = pcaout.fit( grid )
    
    eigenspec   = pcafit.components_.T * pcafit.singular_values_
    eigenspec  /= np.sqrt( pcafit.n_samples_ )
    
    return eigenspec, compsneed

###
    
def calc_w_hat( grid, eigenspec ):
    
    eigen_pinv = np.linalg.pinv( eigenspec )
    
    test = eigenspec.T @ eigenspec
    test[test<1e-15] = 0.0
    test = np.linalg.inv( test )
    
    eigen_pinv = test @ eigenspec.T
        
    w_hat = eigen_pinv @ grid.T
    
    return w_hat, w_hat.reshape( w_hat.size )

### 
    
def createPhiGrid( eigenspec, Nmodels ):
    
    eyemat  = np.eye( Nmodels )
    
    phigrid = []
    
    for i in range( eigenspec.shape[1] ):
                
        phigrid.append( np.kron( eyemat, eigenspec[:,i] ).T )
        
    return np.hstack( phigrid )

###
    
def createK( params, hyperpars ):
    
    a, l = hyperpars
    
    K = np.zeros( ( params.size, params.size ) )
    
    for i in range( params.size ):
        
        for j in range( params.size ):
            
            pardif = params[i] - params[j]
            
            K[i,j] = a ** 2.0 * np.exp( - 0.5 * pardif * l ** 2.0 * pardif )

    return K

###
    
def createSigmaGrid( params, hyperpars, eigenspec ):
        
    covmatarr = []
    
    for i in range( eigenspec.shape[1] ):
        
        covmat = createK( params, hyperpars )

        covmatarr.append( covmat )
        
    return scipy.linalg.block_diag( *covmatarr )

###
    
def createSigmaStar( thetastar, hyperpars, eigenspec ):
    
    a, l = hyperpars
    
    sigmastar = np.diag( a ** 2.0 * np.ones( eigenspec.shape[1] ) )

    return sigmastar

###

def createSigmaGridStar( thetastar, params, hyperpars, eigenspec ):
    
    a, l = hyperpars
    
    sigmagridstar = np.zeros( ( params.size * eigenspec.shape[1], eigenspec.shape[1] ) )
    
    for i in range( params.size ):
        
        pardif = params[i] - thetastar
        
        for j in range( eigenspec.shape[1] ):

            sigmagridstar[i,j] = a ** 2.0 * np.exp( -0.5 * pardif * l ** 2.0 * pardif )

    return sigmagridstar

###
    
def calcInverse4Sigma_w( hyperpars, phigrid, sigmagrid, sigmagridstar ):
    
    l = hyperpars
    
    A = l * ( phigrid.T @ phigrid ) + sigmagrid
    
    return np.linalg.solve( A, sigmagridstar )

###

def calcInverse4mu_w( hyperpars, phigrid, sigmagrid, whatgrid ):
    
    l = hyperpars
    
    A = l * ( phigrid.T @ phigrid ) + sigmagrid
    
    return np.linalg.solve( A, whatgrid )
    
###
    
def calcMuSigma_w( hyperpars, phigrid, sigmagrid, sigmagridstar, whatgrid ):
    
    inv4sig = calcInverse4Sigma_w( hyperpars, phigrid, sigmagrid, sigmagridstar )
    
    inv4mu  = calcInverse4mu_w( hyperpars, phigrid, sigmagrid, whatgrid )
    
    mu_w    = sigmagridstar.T @ inv4mu
    
    sigma_w = sigmastar - ( sigmagridstar.T @ inv4sig )
    
    return mu_w, sigma_w

###
    
def MaternWeight( r, r0 ):
    
    if r <= r0:
        
        return 0.5 + 0.5 * np.cos( np.pi * r / r0 )
    
    else:
    
        return 0.0
    
def calcObsCov( hyperpars, wl ):
    
    a, l = hyperpars
    
    r0 = 4 * l
    
    C = np.zeros( ( wl.size, wl.size ) )
    
    for i in range( C.shape[0] ):
        
        for j in range( C.shape[1] ):
            
            r = 0.5 * 3e5 * np.abs( ( wl[i] - wl[j] ) ) / ( wl[i] + wl[j] )
            
            w = MaternWeight( r, r0 )
            
            C[i,j] =  w * a * ( 1 + np.sqrt(3) * r / l ) * np.exp( - np.sqrt(3) * r / l )
    
    return C
    
def hyperpar_prior():
    
    a_gamma = 1
    b_gamma = 1e-4
    
    return np.random.gamma( a_gamma, b_gamma )

### Stuff for emcee
    
def loglikelihood( x, data, xivals, eigenspec ):
    
    xi_mu, xi_sigma = xivals
    mu_w, sigma_w   = calcMuSigma_w()
    
    C = calcObsCov()
    
    mu    = xi_mu + xi_sigma * ( eigenspec @ mu_w )
    sigma = ( xi_sigma * eigenspec ) @ sigma_w @ ( xi_sigma * eigenspec ).T + C
    
    lnlike = np.log( stats.multivariate_normal( mean = mu, cov = sigma ).pdf( data ) )
    
    return lnlike

def logpriors( hyperpar ):
    
    distr  = stats.gamma( a = 1, scale = 1e-4 )
    
    lnprior = np.log( distr.pdf( hyperpar ) )
    
    return lnprior

flux, wl, temps = makeGridNumpy()

flux_whtnd, mu, sigma = whitenTemplates( flux, return_musig = True )

eigenspec, compsneed = getPCA_Eigenspec( flux_whtnd, varfrac = 0.98 )

w_hat_mat, w_hat_vec = calc_w_hat( flux_whtnd, eigenspec )

phigrid = createPhiGrid( eigenspec, flux.shape[0] )

covgrid = createSigmaGrid( temps, ( 1.0, 1.0 ), eigenspec )

sigmastar = createSigmaStar( 4250, ( 1.0, 1.0 ), eigenspec )

sigmagridstar = createSigmaGridStar( 4250, temps, ( 1.0, 1.0 ), eigenspec )

testmu, testsig = calcMuSigma_w( 1.0, phigrid, covgrid, sigmagridstar, w_hat_vec )






