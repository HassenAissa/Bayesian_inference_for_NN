test:
	python3 package/simple_example.py
venv:
	python3 -m venv venv
activate:
	source venv/bin/activate
install:
	pip install -r requirements.txt