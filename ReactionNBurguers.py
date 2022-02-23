import numpy as np
from numpy.fft import fftn,ifftn,fftfreq
import math
import random
import itertools
import tqdm as tqdm
from collections import namedtuple
from scipy.signal import convolve2d
import cNoise
from scipy.interpolate import interp1d

class ReactionNBurguers():
	'''
	Wrote by: Rubens Andreas Sautter (2021)
	
	Adapted from Huang, et.al.(2019)
	https://advancesindifferenceequations.springeropen.com/articles/10.1186/s13662-019-2328-5
	
	A Noise term can be applied to the advection term: 
	https://github.com/caseman/noise
	
	Perlin Noise:
	Lagae, Ares, et al. "A survey of procedural noise functions." Computer Graphics Forum. Vol. 29. No. 8. Oxford, UK: Blackwell Publishing Ltd, 2010.
	
	Burguers-Bateman alternative equation solved with pseudospectral methods, and integrated with RK45.
	
	du             du        d²u
	__  + s W(t) u __  =  v  ___ + f(u,v)
	dt             dx        dx²
	
	
	dv             dv        d²v
	__  + s W(t) v __  =  v  ___ + g(u,v)
	dt             dx        dx²
	 
	'''

	def __init__(self, alpha=1.13, b=0.1781, e1=0.3, e2=0.2,  mu_v=1.0, mu_u=3.2,s_u=0.1, s_v=0.32, r=0.2,h=6.0, msize = 128,ic='r', mNoise='advection', kappa=0.1, noiseType=None,noiseArgs=None):
		'''
		Spatial parameters:
			ic = initial condition('r', 'g')
			h - grid spacing
			
			mu_u - prey viscosity 
			mu_v - predator viscosity
			s_v - predator spatial velocity - burguers and transport equation hybrid
			s_u - prey spatial velocity - burguers and transport equation hybrid
			gamma - reaction rate
			
		Reaction parameters:
			alpha - hunt rate
			b - predator gain
			e1 - inverse hunt rate
			e2 - inverse predator gain
			r - reaction speed
			
		Noise Parameters:
			mNoise - 'advection' - applies the noise on advection
			       - 'diffusion' - applies the noise on diffusion
			       
			kappa - noise velocity - [0-1]
		
			noiseType - 'gradient' - Perlin noise (https://en.wikipedia.org/wiki/Perlin_noise)
				  - 'colored' - (https://en.wikipedia.org/wiki/Colors_of_noise)
				  - 'fixed' -   gradient noise with the same values at each time step
				  - None - advection in a single direction
				  
			noiseArgs - Gradient noise - {'octaves':32,'persistence':0.5}
				  - Colored noise - {'beta':2} 
				  - None - if none or any of the parameters is specified, the above parameters are applied
		'''

		self.alpha,self.b,self.e1,self.e2,self.r = alpha, b, e1, e2, r
		self.mu_v, self.mu_u, self.s_u, self.s_v,self.h,self.ndim = mu_v, mu_u, s_u, s_v,h,2
		
		
		u0,v0 = 0.5,0.6
		
		u0 = 0.5*(1-alpha-e1+np.sqrt((alpha+e1-1)*(alpha+e1-1)-4*(alpha*e2-e1)))
		v0 = u0+e2
		
		print("Stable (u*,v*): ",u0,v0)
		
		self.ic = ic
		self.u0 ,self.v0 = u0,v0 
		self.msize = msize
		self.dim = 2
		self.noiseType=noiseType
		self.mNoise =mNoise
		self.kappa = kappa
		if noiseArgs is None:
			self.noiseArgs = {}
		else:
			self.noiseArgs = noiseArgs

	def __getRandom(self,n,dim):
		newShape = tuple([n for i in range(dim)])
		return np.random.rand(n**dim).reshape(newShape)
		
	def __getGaussian(self,n,dim):
		out = np.zeros(np.repeat(n,dim))
		c = n/2
		squareDists = np.sum((np.indices(out.shape)-c)**2,as_vs=0)
		return np.exp(-squareDists/n)
		
	def getInitialCondition(self):
		if self.ic=='r':
			self.u = self.u0*(1+1e-2*(self.__getRandom(self.msize,self.dim)-0.5))
			self.v = self.v0*(1+1e-2*(self.__getRandom(self.msize,self.dim)-0.5))
		else:
			self.u = self.u0*(1+1e-2*self.__getGaussian(self.msize,self.dim))
			self.v = self.v0*(1+1e-2*self.__getGaussian(self.msize,self.dim))
		return np.array([self.u, self.v])
		
	
	def getChainedSingleReaction(self,u0=None,v0=None,dt=0.1, nit=3000):
		ul,vl = [],[]
		delta = 1e-6*(np.random.rand()-0.5)
		if u0 is None:
			ut = self.u0+delta
		else:
			ut = u0
		if v0 is None:
			vt = self.v0+delta
		else:
			vt = v0	
		for i in range(nit):
			ul.append(ut)
			vl.append(vt)
			
			t = i*dt
			state = np.array([ut,vt])
			k1 = self.reaction(state[0],state[1],t)
			k2 = self.reaction((state+dt*k1/2)[0],(state+dt*k1/2)[1], t+dt/2.)
			k3 = self.reaction((state+dt*k2/2)[0],(state+dt*k2/2)[1], t+dt/2.)
			k4 = self.reaction((state+dt*k3)[0],(state+dt*k3)[1], t+dt)
			state = state + dt*(k1+2*k2+2*k3+k4)/6.
			ut, vt = state
		return ul, vl
		
	def generatePNoise(self,nit,octaves=64,p=0.7):
		pn = np.zeros((nit,self.msize,self.msize))
		timeRand = np.random.randint(512)
		for z, k  in enumerate(tqdm.tqdm(np.linspace(0,1,nit))):
			for x,i in enumerate(np.linspace(0,1,self.msize)):
    				for y, j in enumerate(np.linspace(0,1,self.msize)):
        				pn[z,x,y] = noise.snoise3(timeRand+k,i,j,octaves=octaves,persistence=p)
		return np.array(pn)

	def reaction(self,u,v,t):
		alpha, e1, r = self.alpha, self.e1,self.r
		b, e2, r = self.b, self.e2,self.r
		
		ul,vl = u,v
		
		u2 = r*(ul*(1-ul)-alpha*vl*ul/(ul+e1))
		v2 = r*b*vl*(1-vl/(ul+e2))
		
		return np.array([u2,v2])
		
	def interpolateNoise(self,time):
		'''
		Linear interpolation of the noise
		
		return the slice of nx and ny at the given time
		'''
		
		t = np.linspace(0,1,self.nx.shape[0])
		p1, p2 = int(np.floor(self.nx.shape[0]*time)),int(np.ceil(self.nx.shape[0]*time))
		
		t1 = t[p1]
		t2 = t[p2]
		mx1 = self.nx[p1]
		my1 = self.ny[p1]
		mx2 = self.nx[p2]
		my2 = self.ny[p2]
		
		if np.abs(t1-t2)<1e-15:
			mx3,my3 = mx1,my1
		else:
			mx3 = np.abs(t1-time)*mx1/(np.abs(t2-t1))+np.abs(t2-time)*mx2/(np.abs(t2-t1))
			my3 = np.abs(t1-time)*my1/(np.abs(t2-t1))+np.abs(t2-time)*my2/(np.abs(t2-t1))
		
		return mx3, my3
		
		
	def solveRKF45(self,dt,ntimes,stepsave,dtTolerace=1e-4):
		state = self.getInitialCondition()
		times = []
		states = [self.getInitialCondition()]	
			
		w = np.array([	[					0,0,0,0,0,0],
				[1/4,					0,0,0,0,0],
				[3/32,9/32,				0,0,0,0],
				[1932/2197,-7200/2197,7296/2197,	0,0,0],
				[439/216,-8,3680/513,-845/4104,	0,0],
				[-8/27, 2,-3544/2565,1859/4104,-11/40,0]
			])
		t = 0.0
		
		if self.noiseType == 'colored':
			if 'beta' in self.noiseArgs:
				exponent = self.noiseArgs['beta']
			else:
				exponent = 2
			if 'std' in self.noiseArgs:
				std = self.noiseArgs['std']
			else:
				std = 0.03
			self.nx = cNoise.cNoise(beta=exponent,shape=(ntimes+2,self.msize,self.msize))
			self.ny = cNoise.cNoise(beta=exponent,shape=(ntimes+2,self.msize,self.msize))
		elif self.noiseType == 'gradient':
			if 'octaves' in self.noiseArgs:
				octaves = self.noiseArgs['octaves']
			else:
				octaves = 16
			if 'persistence' in self.noiseArgs:
				persistence = self.noiseArgs['persistence']
			else:
				persistence = 0.5
			self.nx = self.generatePNoise(nit=ntimes+2,octaves=octaves,p=persistence)
			self.ny = self.generatePNoise(nit=ntimes+2,octaves=octaves,p=persistence)
		else:
			self.nx = np.ones((ntimes+2,self.msize,self.msize))
			self.ny = np.ones((ntimes+2,self.msize,self.msize))
			
		self.maxTime = (ntimes+1)*dt
		self.nx = self.nx/np.max(np.abs(self.nx[:int(self.kappa*self.maxTime)+1]))
		self.ny = self.ny/np.max(np.abs(self.ny[:int(self.kappa*self.maxTime)+1]))
				
		for time in tqdm.tqdm(range(ntimes)):
		
			step = dt
			
			
			k1 = step*self.spatialDerivatives(state,								t		)
			k2 = step*self.spatialDerivatives(state+k1*w[1,0], 		        				t+step/4	)
			k3 = step*self.spatialDerivatives(state+k1*w[2,0]+k2*w[2,1], 					t+3*step/8	)
			k4 = step*self.spatialDerivatives(state+k1*w[3,0]+k2*w[3,1]+k3*w[3,2],    				t+12*step/13	)
			k5 = step*self.spatialDerivatives(state+k1*w[4,0]+k2*w[4,1]+k3*w[4,2]+k4*w[4,3],    		t+step		)
			k6 = step*self.spatialDerivatives(state+k1*w[5,0]+k2*w[5,1]+k3*w[5,2]+k4*w[5,3]+k5*w[5,4],    	t+step/2	)
			
			approach4 = state + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 -k5/5
			approach5 = state + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
			
			error = np.max(np.abs(approach4-approach5))
			if error> dtTolerace:
				step = dt*((dtTolerace/(2*error))**.25)
			
				k1 = step*self.spatialDerivatives(state,								t		)
				k2 = step*self.spatialDerivatives(state+k1*w[1,0], 		        				t+step/4	)
				k3 = step*self.spatialDerivatives(state+k1*w[2,0]+k2*w[2,1], 					t+3*step/8	)
				k4 = step*self.spatialDerivatives(state+k1*w[3,0]+k2*w[3,1]+k3*w[3,2],    				t+12*step/13	)
				k5 = step*self.spatialDerivatives(state+k1*w[4,0]+k2*w[4,1]+k3*w[4,2]+k4*w[4,3],    		t+step		)
				k6 = step*self.spatialDerivatives(state+k1*w[5,0]+k2*w[5,1]+k3*w[5,2]+k4*w[5,3]+k5*w[5,4],    	t+step/2	)
				
				approach4 = state + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 -k5/5
				
			t += step
			state = approach4
			state = state + step*self.reaction(state[0],state[1],t)
			times.append(t)
			if time in stepsave:
				states.append(state)
		return np.array(states), np.array(times)
		
	def spatialDerivatives(self,state,time):
		ul, vl = state
		
		h, mu_u, mu_v, s_u, s_v = self.h,self.mu_u, self.mu_v, self.s_u, self.s_v	
		
		nx, ny = self.interpolateNoise(self.kappa*time/self.maxTime)
		
		#PseudoSpectral approach:
		fx = 2*np.pi*fftfreq(ul.shape[0])
		fy  = 2*np.pi*fftfreq(vl.shape[1])
		
		ftu = fftn(ul)
		ftv = fftn(vl)
		lapU = np.real(ifftn( (-(fx[None,:]**2)-(fy[:,None]**2)) * ftu))
		lapV = np.real(ifftn( (-(fx[None,:]**2)-(fy[:,None]**2)) * ftv))
		
		derUx = np.real(ifftn( fx[None,:]*1j * ftu))
		derUy = np.real(ifftn( fy[None,:]*1j * ftu))
		derVx = np.real(ifftn( fx[None,:]*1j * ftv))
		derVy = np.real(ifftn( fy[None,:]*1j * ftv))
		
		# diffusion:
		if self.mNoise == 'diffusion':
			derUt = (mu_u/(h**2))*(np.power(nx,2)+1e-5)*lapU
			derVt = (mu_v/(h**2))*(np.power(nx,2)+1e-5)*lapV
		else:
			derUt = (mu_u/(h**2))*lapU
			derVt = (mu_v/(h**2))*lapV
		
		# advection:
		if self.mNoise == 'advection':
			#derUt -= s_u*self.u*((nx)*derUx + (ny)*derUy)/h	
			#derVt -= s_v*self.v*((nx)*derVx + (ny)*derVy)/h
			derUt -= s_u*self.u*((nx*np.cos(2*np.pi*ny))*derUx + (nx*np.sin(2*np.pi*ny))*derUy)/h	
			derVt -= s_v*self.v*((ny*np.cos(2*np.pi*nx))*derVx + (ny*np.cos(2*np.pi*nx))*derVy)/h
		else:
			derUt -= s_u*self.u*(derUx + derUy)/h	
			derVt -= s_v*self.v*(derVx + derVy)/h
		
		return np.array([derUt, derVt])

