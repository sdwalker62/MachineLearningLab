TARGET_PATH='docker-stacks/base-notebook'
CUDA_VER=11.3.1
DIST=ubuntu20.04
NEW_BASE=nvidia/cuda:$(CUDA_VER)-cudnn8-runtime-$(DIST)

gpu-build:
	@git submodule update --recursive --remote
	@python3 replace_container.py $(TARGET_PATH) $(NEW_BASE)
	@cd docker-stacks && make build-all OWNER=samuel62
	@docker tag samuel62/base-notebook:latest samuel62/base-lab:cuda_$(CUDA_VER)
	@docker tag samuel62/minimal-notebook:latest samuel62/minimal-lab:cuda_$(CUDA_VER)
	@docker tag samuel62/scipy-notebook:latest samuel62/scipy-lab:cuda_$(CUDA_VER)
	@docker tag samuel62/datascience-notebook:latest samuel62/datascience-lab:cuda_$(CUDA_VER)

dev:
	@pip3 install docker