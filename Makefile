default: pycc

all: ruff mypy pycc

check: ruff mypy

pycc: acropolis/aot
	python3 $^/pycc_cascade.py

ruff: acropolis
	ruff check $^

mypy: acropolis
	mypy $^

clean:
	find . -type f -name *.so -delete
	find . -type f -name *.pyd -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
