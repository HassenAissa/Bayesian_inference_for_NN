test:
	python3 package/simple_example.py
test1:
	python3 package/unittest1.py
test2:
	python3 package/unittest2.py
venv:
	python3 -m venv venv
install:
	pip install -r requirements.txt
gym:
	python3 package/gym_example_1.py