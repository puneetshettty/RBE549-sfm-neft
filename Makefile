.ONESHELL:
SHELL := /bin/bash

phase1/wrapper: venv ## Run Phase 1 Wrapper
	@source venv/bin/activate
	@cd Phase1
	@python Wrapper.py --ImageSet Set1

phase2/train/lego: venv ## Run Phase 1 Wrapper
	@source venv/bin/activate
	@cd Phase2
	@python Wrapper.py --data_path ./data/lego

phase2/wrapper: venv ## Run Phase 1 Wrapper
	@source venv/bin/activate
	@cd Phase2
	@python Wrapper.py 

get-shell: venv ## Get a shell in the virtual environment
	@source venv/bin/activate
	@python

venv: venv/touchfile ## Create a virtual environment

venv/touchfile: requirements.txt
	@test -d venv || python3 -m venv venv
	@. venv/bin/activate; pip install -Ur requirements.txt
	@touch venv/touchfile

clean: ## Clean up the project
	rm -rf venv
	find -iname "*.pyc" -delete

help:: ## Show this help text
	@gawk -vG=$$(tput setaf 2) -vR=$$(tput sgr0) ' \
	  match($$0, "^(([^#:]*[^ :]) *:)?([^#]*)##([^#].+|)$$",a) { \
	    if (a[2] != "") { printf "    make %s%-18s%s %s\n", G, a[2], R, a[4]; next }\
	    if (a[3] == "") { print a[4]; next }\
	    printf "\n%-36s %s\n","",a[4]\
	  }' $(MAKEFILE_LIST)
	@echo  "" # blank line at the end

.DEFAULT_GOAL := help