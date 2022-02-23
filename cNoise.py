from numpy.random import normal
import numpy as np

def cNoise(beta,shape=(1024,),std=0.001):
    '''
       Wrote by: Rubens Andreas Sautter (2021)
       
       Approximating spectral decay by the squared root of the 1/(f^n).
       Frequency are measured in multidimensional space by the frequency euclidian distance
       
       The gaussian standard deviation 
       
       Based on paper:
      http://articles.adsabs.harvard.edu//full/1995A%26A...300..707T/0000707.000.html
    '''
    dimension = []
    for index,dsize in enumerate(shape):
        dimension.append(np.fft.fftfreq(dsize).tolist())
    dimension = tuple(dimension)
    d = float(len(dimension))
    
    freqs = np.power(np.sum(np.array(np.meshgrid(*dimension,indexing='ij'))**2,axis=0),1/2)
    
    #Sampling gaussian with sandard deviation varying according to frequency
    ftSample = normal(loc=0,scale=std,size=shape) + 1j*normal(loc=0,scale=std,size=shape)
    
    # Setting the scale [0,2pi]
    freqs = np.pi*freqs
    not0Freq = (np.abs(freqs)>1e-15)
    
    decayCorrection = np.sqrt(2)**(d-1)

    scaling = (freqs[not0Freq]+0j)**(-(beta*decayCorrection )/2)
    
    ftSample[not0Freq] = (ftSample[not0Freq]*scaling)
    
    not0Freq = (np.abs(freqs)>1e-15)
    
    spsd = np.sum(np.abs(ftSample))
    
    # zero avg
    ftSample[0] = 0.0
   
    out = np.fft.ifftn(ftSample*spsd).real
    	
    return out
