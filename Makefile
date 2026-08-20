PYTHON ?= python3

.PHONY: all data figures validate clean example

all: validate

data:
	$(PYTHON) scripts/build_configurations.py
	$(PYTHON) scripts/coding_reliability.py
	$(PYTHON) scripts/generate_manuscript_csv.py

figures: data
	$(PYTHON) scripts/generate_figures.py

validate: figures
	$(PYTHON) scripts/validate.py

example:
	$(PYTHON) examples/task_sufficiency_illustration/run.py

clean:
	rm -f results/*.csv figures/*.pdf figures/*.png
