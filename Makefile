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