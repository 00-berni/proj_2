import numpy as np
import matplotlib.pyplot as plt
 
def field_image(fig, image, F: np.ndarray, v: int = 0, sct: tuple = (0,None), norm: str = 'linear') -> None:
    """Function to display the field.
    It is possible to display only a section of the field 
    through the parameter `sct` 

    :param fig: figure variable
    :type fig: Any
    :param image: subplot variable
    :type image: Any
    :param F: field matrix
    :type F: np.ndarray
    :param v: set the color of the image: 1 for artificial color, 0 for grayscale, -1 for inverse grayscale. Defaults to 0.
    :type v: int, optional
    :param sct: selected square section of the field, defaults to (0,-1)
    :type sct: tuple, optional
    """ 
    # extracting the edges of image
    a,b = sct
    # setting the color map through `v` param
    if v == -1:  color = 'gray_r' 
    elif v == 0: color = 'gray'
    elif v == 1: color = 'viridis' 
    elif v == 2: color = 'brg'
    # generating the image
    pic = image.imshow(F[a:b,a:b],  origin='lower', cmap=color, norm=norm)
    # adjusting the position and the size of colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(image)
    colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)  
    # generating the colorbar
    fig.colorbar(pic, ax=image, cmap=color, norm=norm, cax=colorbar_axes)

def fast_image(F: np.ndarray, v: int = 0, sct: tuple = (0,None), norm: str = 'linear',title: str = '') -> None:
    fig, ax = plt.subplots(1,1)
    ax.set_title(title,fontsize=20)
    field_image(fig,ax,F,v,sct,norm)
    plt.show()