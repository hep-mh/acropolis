aot: acropolis/aot
	python3 $^/compile.py

ruff: acropolis
	ruff check $^

mypy: acropolis
	mypy $^
