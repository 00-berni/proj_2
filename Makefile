# test dir
TDIR = ./tests
NDIR = ./notebook
RDIR = ./../proj2_res

# To install all the required packages
requirements:
	python3.10 -m pip install -r requirements.txt
# script:
script:
	python3.10 ./script.py > $(RDIR)/script_results.txt
# test:
test:
	python3.10 ./test_rl.py > $(RDIR)/test_results.txt
# test threshold
test-thr:
	python3.10 $(TDIR)/test_thr.py > $(TDIR)/out-test_thr.txt
# jupiter to pdf
notebook-pdf:
	jupyter nbconvert $(NDIR)/implementation_notebook.ipynb --to pdf
	evince $(NDIR)/implementation_notebook.pdf
