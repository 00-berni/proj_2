# FIELD RESTORATION

**Table of Contents**<a id='toc0_'></a> 

- [**Description of the Project**](#toc1_)
    - [**Task**](#toc1_1_)
    - [**Quick commands**](#toc1_2_)
    - [**Project directory overview**](#toc1_3_)
    - [**Notation of comments**](#toc1_4_)
- [**Project tree**](#toc2_)
    - [**skysimulation**](#toc2_1_)
    - [**Makefile**](#toc2_2_)
    - [**script.py**](#toc2_3_)
        - [**Input**](#toc2_3_1)
        - [**Output**](#toc2_3_2)
- [**References**](#toc_)
---

## <a id='toc1_'></a>[Description of the Project](#toc0_)

This project is an university exercise. 

### <a id='toc1_1_'></a>[Task](#toc0_)
The task is to simulate the image restoration of a star sample. Three steps are required:

1. Generate the sample from a Salpeter's IMF
2. Add the background noise, the seeing effect and the noise of the detector
3. Restore the sample from the field

This procedure is implemented by the modules in the `skysimulation` directory, that is the python package. To have more details see the section about [<u>the script</u>](#toc2_4_) and the package.

### <a id='toc1_2_'></a>[Quick commands](#toc0_)

- The list of used python packages and their versions is in the file `requirements.txt`. Hence, one have **to install them to run the script preventing possible errors of compatibility**.
    
    A quick way to do that is to use the command: `$ make requirements`.

- The fastest and safe procedure to run the script is to use the command: `$ make script`.

For other commands see the section about makefile.


### <a id='toc1_3_'></a>[Project directory overview](#toc0_)
In addition to the package (that is the aim of the exercise), the project directory contains:

- a jupiter notebook to store and to explain in details every single step of the implementation (see the notebook section)
- a compilable file to do some operation quickly (see commands section or makefile section)
- a directory for tests (see the test section)

### <a id='toc1_4_'></a>[Notation of comments](#toc0_)

- `#:` is used for definitions
- `#` is used to comment instructions
- `#..` is used to explain an instruction in more lines than one
- `##` is used for the title of a paragraph
- `###` is used for the title of a section


## <a id='toc2_'></a>[Project tree](#toc0_)

- [***skysimulation/***](skysimulation) - the implemented package

    - [`__init__.py`](skysimulation/__init__.py)

    - [`display.py`](skysimulation/display.py) : module to display the field

    - [`field.py`](skysimulation/field.py) : module to initialize the field

    - [`restoration.py`](skysimulation/restoration.py) : module to restore the sample

- [***notebook/***](notebook) - notebook directory

    - [`implementation_notebook.ipynb`](notebook/implementation_notebook.ipynb) : jupyter notebook
    
    - [***Pictures/***](notebook/Pictures) - directory to store the pictures, returned by the code in the notebook

- [***tests/***](tests) - tests directory

- [`.gitignore`](.gitignore)

- [`LICENCE.md`](LICENCE.md) : the free licence

- [`Makefile`](Makefile) : compilable file for useful commands

- [`README.md`](README.md) : informations about the project 

- [`README.txt`](README.txt) : same file as this one in `.txt`

- [`requirements.txt`](requirements.txt) : required packages and versions

- [`script.py`](script.py)

### <a id='toc2_1'></a>[skysimulation](#toc0_)

The package is made up of 3 modules[^1]:

- `display.py`

    This module has the function to display the field and the stars. The aim is to speed up the code writing and to set the same parameters of all the pictures.

- `field.py`

    This module contains all the functions to inizialized the process, such as the generation of the star sample, the location in the field, the kernel function and the procedure to generate noise.

- `restoration.py`

    This module collects functions to extract the objects from the field and to restore them through the R-L procedure (see the articles in [References](#toc)). The implementation of these methods is the main part of the exercise, then I want to say a few more words (for a better and more detailed description see the corrisponding section in the [notebook](notebook/implementation_notebook.ipynb) or the documentation of the module):
    
    1. **extraction**
    
        To extract an object from the field, one must first define what we mean by _object_.  This is completely an arbitrary choice. 
        
        When the sample is generated, an _object_ is simple a not-empty pixel: the whole brightness of a star is contained in a single pixel; but, after the convolution with the PSF, light is spread around that single pixel. That is why for this implementation an _object_ is:
        
        > a generic matrix $n\times m$ of pixels that has the brightest one as its center and a decreasing trend in brightness as one procresses from the center toward the edges. The dimensions of the object are limitated[^2]: $m,n \leq 7$.

        In brief the algorithm for the extraction consists in a routine that stops when a particular threshold value is reached:
        
        1. Find the pixel with the maximum value in brightness
        2. Compute the _signal-to-noise_ ratio (SNR)
        3. If $SNR>threshold$ go to (d.), otherwise go to (g.)
        4. Compute the size of the object according to the previous definition
        5. Store and remove the object
        6. Go to (a.)
        7. Stop  
        
        [^2]: This limit prevents crashes: in fact, if the number of stars is much less than the size of the field ($M\ll N$), then the condition on the trend alone leads to either over-extended objects or a crash.  

    1. **restoration**

        After that all detected object were estracted, the program starts to restore star brightness one by one. The algorithm to do that is an iterative procedure taken from the articles of Lucy and Richardson (see [References]()).


[^1]: the module `__init__.py` is not necessary

### <a id='toc2_2_'></a>[Makefile](#toc0_)

### <a id='toc2_3_'></a>[script.py](#toc0_)

#### <a id='toc2_3_1_'></a>[Inputs](#toc0_)

The script takes :

1. the dimension of the field (that is a matrix $N \times N$)
1. the number $M$ of stars 
1. the kind of seeing function (Gaussian or Lorentzian)
1. the magnitude of both background and detector noises

Due to the convolution with seeing an odd number for $N$ is better.

#### <a id='toc2_3_2_'></a>[Outputs](#toc0_)

The script restores the star sample and shows it in an histogram. The plot contains also the true sample in order to compare the output with the input.  


## [References](#toc0_)

1. L. B. Lucy, _An iterative technique for the rectification of observed distributions_, ApJ, 79:745, June, 1974, doi: [10.1086/111605](https://ui.adsabs.harvard.edu/link_gateway/1974AJ.....79..745L/doi:10.1086/111605).
2. Notes of _Astrofisica Osservativa_ course, 2021-2022.
3. W. H. Richardson, _Bayesian-based iterative method of image restoration_, Journal of the Optical Society of America (1917-1983), 62(1):55, January, 1972, url: [https://ui.adsabs.harvard.edu/abs/1972JOSA...62...55R](https://ui.adsabs.harvard.edu/abs/1972JOSA...62...55R).


