test:
	python3 Pyesian/test_runner.py
venv:
	python3 -m venv venv
install:
	pip install -r requirements.txt

pdoc:
	pydoc --html Pyesian

vis:
	python3 visualize.py

test1:
	python3 Pyesian/tests/unittest1.py

test2:
	python3 Pyesian/tests/unittest2.py