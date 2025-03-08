default: pycc

all: ruff mypy pycc

check: ruff mypy

pycc: acropolis/aot
	python3 $^/pycc_cascade.py

ruff: acropolis
	ruff check $^

mypy: acropolis
	mypy $^

clean: acropolis/aot
	rm -rf $^/*.so
