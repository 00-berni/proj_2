# To install all the required packages
requirements:
	python3.10 -m pip install -r requirements.txt
# default
default:
	python3 ./default_field.py
# storing default
s_default:
	python3 ./default_field.py > results.txt
# multi run
multi:
	python3 ./multiple_runs.py
# storing multi
s_multi:
	python3 ./multiple_runs.py > multi_results.txt
# poisson
poisson:
	python3 ./poisson.py
# store poisson
poisson:
	python3 ./poisson.py > poisson_results.txt

