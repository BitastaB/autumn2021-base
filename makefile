.PHONY: help
.PHONY: conda-env reset

.DEFAULT: help

help:
	@echo "conda-env"
	@echo "        Create conda environment"
	@echo "reset"
	@echo "        Remove and reinstall conda environment"


conda-env:
	@conda env create --file environment.yml

reset:
	@conda env remove --name autumn2021
	@make conda-env
