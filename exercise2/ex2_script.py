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
from functions import *
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

# %% [markdown]
# ### Fifth cell
# I implemented the initialization function to generate the masses, the field and the image without any psf or noise.

# %%


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

# fig1.savefig('./Pictures/field.png')
# fig2.savefig('./Pictures/zoom.png')

plt.show()
	
	

# %% [markdown]
# ### Seventh cell
# I implemented function to add noise, both background and detector noise.

# %%
# SEVENTH CELL



# generation of the field and the stars
F, S = initialize()

# add background noise
F_n = F + noise()

## Plot
fig1, ((img_field, img_field_noise),(img_field_seeing,img_field_snoise)) = plt.subplots(2,2,figsize=(17,17))
fig2, ((img_zoom, img_zoom_noise)  ,(img_zoom_seeing,img_zoom_snoise))   = plt.subplots(2,2,figsize=(17,17))
fig3, (img_field_tot, img_zoom_tot) = plt.subplots(2,1,figsize=(12,20))

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


# fig1.savefig('./Pictures/field_noise.png')
# fig2.savefig('./Pictures/zoom_noise.png')
# fig3.savefig('./Pictures/image.png')

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


