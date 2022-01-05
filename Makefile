# remember cat -e -t -v Makefile to check for hard tabs

SOURCE_PATH='docker-stacks/base-notebook'
CUDA_VER:=11.3.1
DIST:=ubuntu20.04
NEW_BASE:=nvidia/cuda:$(CUDA_VER)-cudnn8-runtime-$(DIST)

OWNER:=samuel62

LAB_LIST:= \
base \
minimal \
scipy \
datascience

CUSTOM_LAB_LIST:= \
mll \
rll

ALL_LIST:= \
base \
minimal \
scipy \
datascience \
mll \
rll


gpu-build:
	@git submodule update --recursive --remote
	@python3 replace_container.py $(SOURCE_PATH) $(NEW_BASE)
	@cd docker-stacks && make build-all OWNER=samuel62
	@docker tag samuel62/base-notebook:latest samuel62/machine_learning_lab:base_cuda_$(CUDA_VER)
	@docker tag samuel62/minimal-notebook:latest samuel62/machine_learning_lab:minimal_cuda_$(CUDA_VER)
	@docker tag samuel62/scipy-notebook:latest samuel62/machine_learning_lab:scipy_cuda_$(CUDA_VER)
	@docker tag samuel62/datascience-notebook:latest samuel62/machine_learning_lab:data_science_cuda_$(CUDA_VER)


install-dependencies:
	@pip3 install docker
	@pip3 install tqdm
	@pip3 install -U pytest
	@pip3 install black
	@pip3 install tabulate


test/%: ## run tests for each image
	@python3 tests/exec_tests.py $(notdir $@) $(OWNER) $(CUDA_VER)
test-all: $(foreach I, $(ALL_LIST), test/$(I)) ## test all docker-stack images


docs/%: ## generate documentation for each image
	@python3 utils/generate_docs.py $(OWNER)/machine_learning_lab:$(notdir $@)_cuda_$(CUDA_VER)
docs-all: $(foreach I, $(LAB_LIST), docs/$(I)) ## generate all docs


format:
	@black --verbose --exclude=docker-stacks .