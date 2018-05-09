from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

import h5py, scipy.linalg, pdb

### Function to turn the hdf5 grid output from Starfish itself into a numpy array
#   Also returns the wavelength array and an array of temps corresponding to each entry in the template grid

def makeGridNumpy():
    
    gridIN     = h5py.File( 'grid.hdf5', 'r' )
    
    model_wl   = np.array( gridIN['wl'] )
    model_flux = np.zeros( ( gridIN['pars'].shape[0], model_wl.size ) )
    model_Teff = np.transpose( gridIN['pars'] )[0]
    
    for i in range( model_Teff.size ):
        
        modstr        = 't' + str( int( model_Teff[i] ) ) + 'g4.5Z0.0'
        
        model_flux[i] = np.array( gridIN['flux'][modstr] )

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
    
def createCovK( params, hyperpars ):
    
    a, l = hyperpars
    
    K = np.zeros( ( params.size, params.size ) )
    
    for i in range( params.size ):
        
        for j in range( params.size ):
            
            pardif = params[i] - params[j]
            
            K[i,j] = a ** 2.0 * np.exp( - 0.5 * pardif * l ** 2.0 * pardif )

    return K

###
    
def createCovGrid( params, hyperpars, eigenspec ):
        
    covmatarr = []
    
    for i in range( eigenspec.shape[1] ):
        
        covmat = createCovK( params, hyperpars )

        covmatarr.append( covmat )
        
    return scipy.linalg.block_diag( *covmatarr )

###
    
def createCovStar( thetastar, params, hyperpars ):
    
    a, l = hyperpars
    
    K = np.zeros( )
    
def hyperpar_prior():
    
    a_gamma = 1
    b_gamma = 1e-4
    
    return np.random.gamma( a_gamma, b_gamma )

flux, wl, temps = makeGridNumpy()

flux_whtnd, mu, sigma = whitenTemplates( flux, return_musig = True )

eigenspec, compsneed = getPCA_Eigenspec( flux_whtnd, varfrac = 0.98 )

w_hat_mat, w_hat_vec = calc_w_hat( flux_whtnd, eigenspec )

phigrid = createPhiGrid( eigenspec, flux.shape[0] )

covgrid = createCovGrid( temps, ( 1.0, 1.0 ), eigenspec )







