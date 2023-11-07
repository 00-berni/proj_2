import numpy as np
import matplotlib.pyplot as plt
 
def field_image(fig, image, F: np.ndarray, v: int = 0, sct: tuple = (0,-1), norm: str = 'log') -> None:
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
    if v == 0: color = 'gray'
    elif v == 1: color = 'viridis' 
    else: color = 'gray_r' 
    # generating the image
    pic = image.imshow(F[a:b,a:b], cmap=color, norm=norm)
    # generating the colorbar
    fig.colorbar(pic, ax=image, cmap=color, norm=norm, location='bottom')

def fast_image(F: np.ndarray, v: int = 0, sct: tuple = (0,-1), norm: str = 'log') -> None:
    fig, ax = plt.subplots(1,1)
    field_image(fig,ax,F,v,sct,norm)
    plt.show()