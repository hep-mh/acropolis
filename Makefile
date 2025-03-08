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

upload-test:
	twine check dist/*
	twine upload --repository testpypi dist/*

clean-pycc:
	find . -type f -name *.so -delete
	find . -type f -name *.pyd -delete

clean-build:
	rm -rf dist/ build/ ACROPOLIS.egg-info/

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean: clean-pycc clean-cache clean-build
