test:
	python3 package/simple_example.py
venv:
	python3 -m venv venv
install:
	pip install -r requirements.txt
gym:
	python3 package/gym_example_1.py