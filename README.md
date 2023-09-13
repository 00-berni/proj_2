# FIELD RESTORATION

**Table of Contents**<a id='toc0_'></a> 

- [**Description of the Project**](#toc1_)
    - [**Task**](#toc1_1_)
    - [**Quick commands**](#toc1_2_)
    - [**Project directory overview**](#toc1_3_)
- [**Project tree**](#toc2_)
    - [**Makefile**](#toc2_1_)

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

### <a id='toc2_1_'></a>[Makefile](#toc0_)



### <a id='toc2_4_'></a>[script.py](#toc0_)

#### <a id='toc1_1_'></a>[Inputs](#toc0_)

The script takes :

1. the dimension of the field (that is a matrix $N \times N$)
1. the number $M$ of stars to generate 
1. the kind of seeing function (Gaussian or Lorentzian)
1. the magnitude of both background and detector noises

Due to the convolution with seeing a odd number for $N$ is better.

#### <a id='toc1_2_'></a>[Outputs](#toc0_)

The script returns 


## [References](#toc0_)