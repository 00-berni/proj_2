
# VISIBILITY OF A STAR

**Table of Contents**<a id='toc0_'></a> 

- [**Description of the Project**](#toc1_)
- [**Project tree**](#toc2_)
    - [**Makefile**](#toc2_1_)


---

## <a id='toc1_'></a>[Description of the Project](#toc0_)

This project is an university exercise. 

The `skysimulation` directory is the python package of functions for the visibility of a star: set a place from which to observe and the date of the observation, it is possible to compute if a target object is visible or not. To have more details see the section about [<u>the script</u>](#toc2_4_).

In addition to this package (that is the aim of the exercise), the project directory contains:

- a jupiter notebook to store and explain in details every single step of the implementation 
- 



## <a id='toc2_'></a>[Project tree](#toc0_)

- [`README.md`](README.md) : informations about the project 

- [`README.txt`](README.txt) : same file as this one in `.txt`

- [`Makefile`](Makefile) : compilable file

- [`requirements.txt`](requirements.txt) : require packages

- [`script.py`](script.py) : the script

- [***skysimulation/***](skysimulation) - the package

    - [`__init__.py`](skysimulation/__init__.py) : script

    - [`display.py`](skysimulation/display.py) : function script

    - [`field.py`](skysimulation/field.py) : function script

    - [`recovery.py`](skysimulation/recovery.py) : function script

- [***notebook/***](notebook) - folder for implementation

    - [`implementation_notebook.ipynb`](notebook/implementation_notebook.ipynb) : jupyter notebook
    
    - [***Pictures/***](notebook/Pictures) - folder with pictures, made by script

- [***tests/***](tests) - tests directory



### <a id='toc2_1_'></a>[Makefile](#toc0_)

Use `proj_2$ make requirements` to install all the necessary packages.


### <a id='toc2_4_'></a>[script.py](#toc0_)

#### <a id='toc1_1_'></a>[Inputs](#toc0_)

The script takes **3 main inputs**: 

1. the **coordinates** and the **proper motion** informations of the star
2. the **coordinates** and the **height** of the observatory location on the Earth
3. the **date** of the observation

#### <a id='toc1_2_'></a>[Outputs](#toc0_)

The script returns 
