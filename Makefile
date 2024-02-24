test1:
	python3 PyAce/unittest1_runner.py
test2:
	python3 PyAce/unittest2_runner.py
venv:
	python3 -m venv venv
install:
	pip install -r requirements.txt

pdoc:
	pydoc --html PyAce

vis:
	python3 visualize.py
