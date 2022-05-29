# %% [markdown]
# # Test Function
# ## I part: generate the field
# ### First cell
# I defined the parameters like the dimension of the field, the number of stars, the exponents of the potential laws and the IMF function

# %%
# FIRST CELL
##* packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# dimension of the matrix
N = int(4e2+1)
# number of stars
M = int(4e2)

## Set parameters
alpha = 2
beta  = 3
# minimum and maximum masses 
m_min = 0.1; m_max = 20
# Initial Mass Function
IMF = lambda m : m**(-alpha)
# def IMF(m):
# 	if(m >= 0.01 and m < 0.08):
# 		a = -0.3
# 	elif(m >= 0.08 and m < 0.5 ):
# 		a = -1.3
# 	else:
# 		a = -2.3
# 	return m**a
IMF_min = IMF(0.1); IMF_max = IMF(20) 

print(f'IMF for smallest and biggest stars:\n0.1 Msun\t{IMF_min}\n20  Msun\t{IMF_max}')


# %% [markdown]
# ### Second and Third cells
# I defined the function for the random generation of the masses and then I created the luminosity array

# %%
# SECOND CELL
##* Genration of masses array with IMF distribution
#   The function takes the minimum and the maximum of the
#   IMF, generates a M-dimension array of random value for imf in
#   [IMF_min,IMF_max] and returns a M-dimension array of masses,
#   distribuited like the IMF   
#	param	min	minimum imf value
# 	param	max	maximum	imf value
#	param	sdim	number of stars. Set to M
# 
# 	return	m	{dim} array of masses distributed like imf   
def generate_mass_array(imf_min,imf_max,sdim=M):
	np.random.seed()
	# imf_trs = IMF(0.5)
	#dim = int(sdim/2)
	# imf1 = np.random.rand(dim)*(imf_trs-imf_max)+imf_max
	# imf1 = imf1**(-1/2)
	# imf2 = np.random.rand(dim)*(imf_min-imf_trs)+imf_min
	# imf2 = imf2**(-1/2)
#	return np.append(imf1,imf2)
	return 	(np.random.rand(sdim)*(imf_min-imf_max)+imf_max)**(-1/alpha)

# M array of masses in solar mass unit
m = generate_mass_array(IMF_min,IMF_max)

## Plot data

plt.figure(figsize=(12,8))
plt.title(f'Mass distribution with $\\alpha = {alpha}$ and {M} stars')
plt.hist(m,int(3*M/5),histtype='step')
plt.xscale('log')
#plt.xlim(0.09,20)
plt.xlabel('m [$M_\odot$]')
plt.ylabel('counts')


plt.show()



# %%
# THIRD CELL
# M array of luminosities in solar luminosity unit
L = m**beta

## Plot data
plt.figure(figsize=(12,8))
plt.title(f'Luminosity distribution with $\\beta = {beta}$ for {M} stars')
plt.hist(np.log10(L),int(2*M/5),histtype='step')
#plt.xlim(np.log10(0.09)*beta,np.log10(20)*beta)
plt.xlabel('$\log{(L)}$ [$L_\odot$]')
plt.ylabel('counts')

plt.show()

# %% [markdown]
# ### Fourth cell
# I defined the star class object, a class that contains all infos about star (mass,lum and position) and I implemented the function to set stars positions and the function to update the field matrix with the stars luminosities in mag. 
# Since the function `plt.imshow()` takes float values from 0 to 1 to generate an image in grayscale, I decided to write the `transfer_function()`, that transforms luminosities into values in [0,1].

# %%
# FOURTH CELL
##* Star class obj to collect star informations
class star(object):
#		param	mass	star mass
#		param	lum	star luminosity
# 		param	x,y 	star coordinates in the field 	
	def __init__(self,mass, lum, x ,y):
		self.m   = mass		# star mass value
		self.lum = lum		# star luminosity value
		self.x = x		# star x coordinate
		self.y = y		# star y coordinate

##* Function to locate the stars
#   It generates 2 random arrays of dimension n: 
#   one is the x coordinate array and 
#   y coordinate array of each star
#	param	sdim	number of stars. Set to M
#	param	dim	dimension of the field. Set to N
#
#
#	return	X,Y	stars coordinate arrays
#? I have to check this function: the condition replace=False forbids M > N  
def star_location(sdim=M,dim=N):
	tmp = np.random.default_rng()
	X = tmp.choice(dim, size=sdim)
	Y = tmp.choice(dim, size=sdim, replace=False)
	return X, Y	



##* Function to update the field
#   It adds the generated stars to the field
#   The shape of the field matrix is discussed
#   in the next cell
#	param	F	field matrix [dim,dim]
#	param	X,Y	coordinate arrays
#	param	l	luminosity array 
# 
# 	return	F	updated field matrix [dim,dim]
def update_field(F,X,Y,l):
	F[X,Y] = l
	return F



##* Transfer function
#?  Understand if it is reasonable 
'''   
	The implementation of this function is my choice. The function plt.imshow() takes float values in [0,1] to make grayscale image.
	I decided to convert luminosities into values in [0,1], using logaritmic values of them: 
		- I set a minimum detectable value of luminosity (called inf)
		  Luminosities < inf are represented as black pixels
		- I normaled with the maximum value of luminosity (called sup)
		  Luminosities = sup are converted in 1 (namely white)
'''
#   It converts luminosities in a scale from 0 to 1 to get grayscale
#   with function plt.imshow()
# 	param	l	array of luminosities
# 
# 	return	mag	converted magnitudes array   
def transfer_function(l):
	inf = 9e-5	# threshold to detect luminosity
	sup = 8000	# maximum luminosity
	# control the presence of 0s, change them with inf and save in an other temporary array
	tmp = np.where(l<=0,inf,l)
	# convertion
	mag = (np.log10(tmp)-np.log10(inf)) / (np.log10(sup)-np.log10(inf))
	# control the presence of negative values in mag and change them with 0
	mag = np.where(mag<0,0,mag); mag = np.where(mag>1,1,mag)
	return mag


##* Function to represent the field
#   It builts a [n,n,3] tensor (called field) and
#   converts the luminosities in [0,1] values
#   with transfer_function()   
#	param	image	image type variable
#	param	F	field matrix
#	param	v	a parameter to set the view
#			if v = -1 the image is in negative
#			set to 0 by default
#	param	sct	selected square section of the field
# 			set to [0,-1] by default
# 
# 	return	void	field is saved in image variable	
def field_image(image,F,v=0,sct=[0,-1]):
	n = len(F)
	field = np.zeros([n,n,3])
	lum = transfer_function(F)
	lum = v*(lum-1) + (1+v)*lum
	for i in range(3):
		field[:,:,i] = lum
	a,b = sct
	image.imshow(field[a:b,a:b,:])
	return 

# %% [markdown]
# ### Fifth cell
# I implemented the initialization function to generate the masses, the field and the image without any psf or noise.

# %%
# FIFTH CELL

##* Initialization function: generation of the "perfect" sky
#   It generates the stars and initialized the field to make
#   the sky image without any psf and noise
#	param	dim	dimension of the field. Set to N
#	param	sdim	number of stars. Set to M
#	
#	return	F	field matrix
#	return	S	star class obj with stars infos
def initialize(dim=N,sdim=M):
	# generate a [N,N,3] matrix of 0s
	# the third index is use for the RGB
	F = np.zeros([dim,dim])
	# generate masses
	m = generate_mass_array(IMF_min,IMF_max)
	# set luminosities
	L = m**beta
	# generate stars coordinates
	xs,ys = star_location(sdim)
	# put stars in the field
	F = update_field(F,xs,ys,L)
	# save stars infos
	S = star(m,L,xs,ys)
	return F, S

# generation of the field and the stars
F, S = initialize()

## Plot
# variables to zoom a sector [inf:sup] x [inf:sup] of the field
inf = int(0.2*N)
sup = int(0.7*N)

fig, (img_field, img_zoom) = plt.subplots(2,1,figsize=(6,15))

#  0 : positive image
# -1 : negative image
v = 0

field_image(img_field,F,v)
#img_field.imshow(1-F)
img_field.set_title(f'Source field with {M} stars')
field_image(img_zoom,F,v,[inf,sup])
#img_zoom.imshow(1-F[inf:sup,inf:sup,:])
img_zoom.set_title(f'Source field with {M} stars: sector [{inf}:{sup}] x [{inf}:{sup}]')
plt.show()

# %% [markdown]
# ### Sixth cell
# I implemented the function for the atmosferic seeing by gaussian distribution

# %%
# SIXTH CELL
##* Gaussian matrix generator
#   It makes a gaussian [n,n] matrix, centered in [xs,ys]
#	param	sigma	the root of variance. Set to 0.9
# 	param	dim	dimension of matrix field. Set to N
#
#	return	G[n,n]	gaussian [n,n] matrix
def gaussian(sigma=0.6,dim=N):
	x = np.arange(dim,dtype=int)
	y = np.arange(dim,dtype=int)
	# shift to center of the field
	x -= int(dim/2);  y -= int(dim/2)
	# Gaussian function
	G = lambda r : np.exp(-r**2/sigma**2/2)
	# generate [n,n] matrix = G_i * G_j
	return np.outer(G(x),G(y))


##* Atmosferic seeing function
#   It convolves the field with tha Gaussian to
#   make the atmosferic seeing
#	param	f	field matrix [n,n]
#	
#	return	f_s	field matrix [n,n] with seeing
#?  Add a parameter to choose atm pfs between Gaussian and Lorentzian
def atm_seeing(f):
	# dim of the field
	n = len(f)
	# call f_s the new field with seeing
#	f_s = f
	# take [n,n] matrix from the field
	field = f
	# convolution with gaussian
	field = fftconvolve(field,gaussian(dim=n),mode='same')
	# values are saved in each color channel 
	# to have grayscale
	return field

# generation of the field and the stars
F, S = initialize()

## Plot
fig1, (img_field, img_field_seeing) = plt.subplots(2,1,figsize=(6,15))
fig2, (img_zoom, img_zoom_seeing)   = plt.subplots(2,1,figsize=(6,15))

#  0 : positive image
# -1 : negative image
v = 0

field_image(img_field,F,v)
#img_field.imshow(1-F)
img_field.set_title(f'No Seeing Field with {M} stars')
field_image(img_zoom,F,v,[inf,sup])
#img_zoom.imshow(1-F[inf:sup,inf:sup,:])
img_zoom.set_title(f'No Seeing Field with {M} stars: [{inf}:{sup}] x [{inf}:{sup}]')

# generation of the seeing image
F_s = atm_seeing(F)
field_image(img_field_seeing,F_s,v)
#img_field_seeing.imshow(1-F_s)
img_field_seeing.set_title(f'Seeing Field with {M} stars')
field_image(img_zoom_seeing,F_s,v,[inf,sup])
#img_zoom_seeing.imshow(1-F_s[inf:sup,inf:sup,:])
img_zoom_seeing.set_title(f'Seeing Field with {M} stars: [{inf}:{sup}] x [{inf}:{sup}]')

fig1.savefig('./Pictures/field.png')
fig2.savefig('./Pictures/zoom.png')

plt.show()
	
	

# %% [markdown]
# ### Seventh cell
# I implemented function to add noise, both background and detector noise.

# %%
# SEVENTH CELL

##* Noise generator
#   It generates a [N,N,3] matrix of noise, using
#   an arbitrary value n (set to 2 * 10**-4) times
#   a random value in [0,1]
#	param	n	value of noise, set to 2e-4
#	param	dim	dimension of the field. Set to N
#
#	return	Noise	[dim,dim] matrix of noise
def noise(n = 2e-4,dim=N):
	np.random.seed()
	# random multiplicative [N,N] matrix
	N0 = np.random.random((dim,dim))*n
	return N0


# generation of the field and the stars
F, S = initialize()

# add background noise
F_n = F + noise()

## Plot
fig1, ((img_field, img_field_noise),(img_field_seeing,img_field_snoise)) = plt.subplots(2,2,figsize=(17,17))
fig2, ((img_zoom, img_zoom_noise)  ,(img_zoom_seeing,img_zoom_snoise))   = plt.subplots(2,2,figsize=(17,17))
fig3, (img_field_tot, img_zoom_tot) = plt.subplots(2,1,figsize=(8,17))

#  0 : positive image
# -1 : negative image
v = 0

field_image(img_field,F,v)
#img_field.imshow(1-F)
img_field.title.set_text(f'No Seeing and noise sky with {M} stars')
field_image(img_zoom,F,v,[inf,sup])
#img_zoom.imshow(1-F[inf:sup,inf:sup,:])
img_zoom.title.set_text(f'No Seeing and noise sky with {M} stars: [{inf}:{sup}] x [{inf}:{sup}]')


field_image(img_field_noise,F_n,v)
#img_field_noise.imshow(1-F_n)
img_field_noise.title.set_text(f'No Seeing Field with {M} stars and sky noise')
field_image(img_zoom_noise,F_n,v,[inf,sup])
#img_zoom_noise.imshow(1-F_n[inf:sup,inf:sup,:])
img_zoom_noise.title.set_text(f'No Seeing Field with {M} stars and sky noise: [{inf}:{sup}] x [{inf}:{sup}]')

# generate atmosferic seeing image without sky noise
F_s = atm_seeing(F)
field_image(img_field_seeing,F_s,v)
#img_field_seeing.imshow(1-F_s)
img_field_seeing.title.set_text(f'Seeing Field with {M} stars without noise')
field_image(img_zoom_seeing,F_s,v,[inf,sup])
#img_zoom_seeing.imshow(1-F_s[inf:sup,inf:sup,:])
img_zoom_seeing.title.set_text(f'Seeing Field with {M} stars without noise: [{inf}:{sup}] x [{inf}:{sup}]')

# generate atmosferic seeing image with sky noise
F_sn = atm_seeing(F_n)
field_image(img_field_snoise,F_sn,v)
#img_field_snoise.imshow(1-F_sn)
img_field_snoise.title.set_text(f'Seeing Field with {M} stars with sky noise')
field_image(img_zoom_snoise,F_sn,v,[inf,sup])
#img_zoom_snoise.imshow(1-F_sn[inf:sup,inf:sup,:])
img_zoom_snoise.title.set_text(f'Seeing Field with {M} stars with sky noise: [{inf}:{sup}] x [{inf}:{sup}]')

# add detector noise, set to 3e-4 (> than background one)
F_sn += noise(3e-4)
field_image(img_field_tot,F_sn,v)
#img_field_tot.imshow(1-F_sn)
img_field_tot.title.set_text(f'Seeing Field with {M} stars with sky and detector noise')
field_image(img_zoom_tot,F_sn,v,[inf,sup])
#img_zoom_tot.imshow(1-F_sn[inf:sup,inf:sup,:])
img_zoom_tot.title.set_text(f'Seeing Field with {M} stars with sky and detector noise: [{inf}:{sup}] x [{inf}:{sup}]')


fig1.savefig('./Pictures/field_noise.png')
fig2.savefig('./Pictures/zoom_noise.png')
fig3.savefig('./Pictures/image.png')

plt.show()


# %% [markdown]
# ## II part: detect stars
# ### First cell

# %%
## First cell

# Dark
dark = noise(3e-4)

d = dark.mean()

print(d)

# Background
bg = atm_seeing(noise())
bg += noise(3e-4)

#bg-=dark

## Plot
fig1, (dark_img, dark_zoom) = plt.subplots(1,2,figsize=(15,7))
fig2, (bg_img, bg_zoom) = plt.subplots(1,2,figsize=(15,7))

#  0 : positive image
# -1 : negative image
v = 0


dark_img.set_title('Dark')
field_image(dark_img,dark,v)
#dark_img.imshow(dark)
dark_zoom.set_title(f'Dark: [{inf}:{sup}] x [{inf}:{sup}]')
field_image(dark_zoom,dark,v,[inf,sup])
#dark_zoom.imshow(dark[inf:sup,inf:sup,:])
bg_img.set_title('background')
field_image(bg_img,bg,v)
#bg_img.imshow(bg)
bg_zoom.set_title(f'Background: [{inf}:{sup}] x [{inf}:{sup}]')
field_image(bg_zoom,bg,v,[inf,sup])
#bg_zoom.imshow(bg[inf:sup,inf:sup,:])

plt.show()	


