# test dir
TDIR = ./tests
NDIR = ./notebook

# To install all the required packages
requirements:
	python3 -m pip install -r requirements.txt
#tests:
script:
	python3 ./exercise2/ex2_script.py
# test threshold
test-thr:
	python3 $(TDIR)/test_thr.py > $(TDIR)/out-test_thr.txt
# jupiter to pdf
notebook-pdf:
	jupyter nbconvert $(NDIR)/implementation_notebook.ipynb --to pdf
	evince $(NDIR)/implementation_notebook.pdf
