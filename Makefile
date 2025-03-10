default: pycc

pycc: 
	python3 acropolis/aot/pycc_cascade.py

check:
	ruff check acropolis
	mypy acropolis

build:
	python3 -m build
#python3 setup.py sdist bdist_wheel
	
upload:
	twine check dist/*
	twine upload --repository pypi dist/*

upload-test:
	twine check dist/*
	twine upload --repository testpypi dist/*

pycc-clean:
	find . -type f -name *.so -delete
	find . -type f -name *.pyd -delete

build-clean:
	rm -rf dist/ build/ ACROPOLIS.egg-info/

cache-clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean: pycc-clean cache-clean build-clean
