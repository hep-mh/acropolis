default: pycc

pycc: 
	python3 acropolis/aot/pycc_cascade.py

check:
	ruff check acropolis
	mypy acropolis

build:
	python3 setup.py sdist bdist_wheel
	
upload:
	twine check dist/*
	twine upload --repository pypi dist/*

upload_test:
	twine check dist/*
	twine upload --repository testpypi dist/*

clean: clean_pycc clean_cache clean_build

clean_pycc:
	find . -type f -name *.so -delete
	find . -type f -name *.pyd -delete

clean_build:
	rm -rf dist/ build/ ACROPOLIS.egg-info/

clean_cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
