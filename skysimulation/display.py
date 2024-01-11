from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
 
def field_image(fig: Figure, image: Axes, F: np.ndarray, v: int = 0, sct: Sequence[int | None | Sequence[int | None]] = (0,None), norm: str = 'linear', colorbar: bool = True, ticks: bool = True) -> None:
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
    if not isinstance(sct[0],(tuple,list)):
        if sct[0] == 0 and sct[1] is None: ticks = False
        sct = [[*sct],[*sct]]
    elif len(sct) == 0 or len(sct[0]) == 0:
        sct = ((None,None),(None,None))
        ticks = False
    xcut = slice(*sct[0])
    ycut = slice(*sct[1])
    # setting the color map through `v` param
    if v == -1:  color = 'gray_r' 
    elif v == 0: color = 'gray'
    elif v == 1: color = 'viridis' 
    elif v == 2: color = 'brg'
    # generating the image
    pic = image.imshow(F[xcut,ycut],  origin='lower', cmap=color, norm=norm)
    if ticks:
        x0,x1 = sct[1]
        y0,y1 = sct[0]
        if x0 is None: x0 = 0
        if y0 is None: y0 = 0
        if x1 is None: x1 = F.shape[0]
        if y1 is None: y1 = F.shape[1]
        xtick = np.arange(x0,x1,(x1-x0)//4)
        ytick = np.arange(y0,y1,(y1-y0)//4)
        image.set_xticks(np.arange(0,x1-x0,(x1-x0)//4))
        image.set_yticks(np.arange(0,y1-y0,(y1-y0)//4))
        image.set_xticklabels(xtick)
        image.set_yticklabels(ytick)
    # adjusting the position and the size of colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(image)
    if colorbar:
        colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)  
        # generating the colorbar
        fig.colorbar(pic, ax=image, cmap=color, norm=norm, cax=colorbar_axes)

def fast_image(F: np.ndarray, title: str = '', **kwargs) -> None:
    fig, ax = plt.subplots(1,1)
    ax.set_title(title,fontsize=20)
    field_image(fig,ax,F,**kwargs)
    plt.show()