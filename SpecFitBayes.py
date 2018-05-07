from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

import h5py, pdb

def makeGridNumpy():
    
    gridIN     = h5py.File( 'grid.hdf5', 'r' )
    
    model_wl   = np.array( gridIN['wl'] )
    model_flux = np.zeros( ( gridIN['pars'].shape[0], model_wl.size ) )
    model_Teff = np.transpose( gridIN['pars'] )[0]
    
    for i in range( model_Teff.size ):
        
        modstr        = 't' + str( int( model_Teff[i] ) ) + 'g4.5Z0.0'
        
        model_flux[i] = np.array( gridIN['flux'][modstr] )
        
    return model_flux, model_wl, model_Teff

def whitenTemplates( models ):
    
    template_mean = np.sum( models, axis = 0 ) / models.size
    template_std  = np.sqrt( np.sum( ( models - template_mean ) ** 2.0, axis = 0 ) )
    
    models_whtnd  = ( models - template_mean ) / template_std
    
    return models_whtnd

def getPCA( flux ):
    
    pcatest    = PCA( n_components = 30 )
    pcafittest = pcatest.fit( flux )

    compsneed  = np.where( np.cumsum( pcafittest.explained_variance_ratio_ ) >= 0.98 )[0][0]
    
    pcaout     = PCA( n_components = compsneed + 1 )
    pcafit     = pcaout.fit( flux )
    
    return pcafit

def constructEigenspec( pca ):
    
    eigenspec  = pca.components_.T  / pca.singular_values_
    eigenspec /= np.sqrt( pca.n_samples_ )
    
    return eigenspec

def hyperpar_prior():
    
    a_gamma = 1
    b_gamma = 1e-4
    
    return np.random.gamma( a_gamma, b_gamma )

flux, wl, temps = makeGridNumpy()

flux_whtnd = whitenTemplates( flux )

flux_pca   = getPCA( flux_whtnd )

eigenspec  = constructEigenspec( flux_pca )








