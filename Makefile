# test dir
TDIR = ./tests

# To install all the required packages
requirements:
	python3 -m pip install -r requirements.txt
#tests:
script:
	python3 ./exercise2/ex2_script.py
# test threshold
test-thr:
	echo - TEST FOR THE STOP IN SEARCHING -
	echo run the script: test_thr.py
	python3 $(TDIR)/test_thr.py > $(TDIR)/out-test_thr.txt
# jupiter to pdf
notebook-pdf:
	jupyter nbconvert ./notebook/implementation_notebook.ipynb --to pdf
	evince ./notebook/implementation_notebook.pdf
