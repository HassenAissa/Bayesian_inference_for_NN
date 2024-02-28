test:
	python3 PyAce/test_runner.py
venv:
	python3 -m venv venv
install:
	pip install -r requirements.txt

pdoc:
	pydoc --html PyAce

vis:
	python3 visualize.py

test1:
	python3 PyAce/tests/unittest1.py

test2:
	python3 PyAce/tests/unittest2.py