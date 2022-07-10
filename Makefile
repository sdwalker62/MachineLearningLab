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
mll

ALL_LIST:= \
base \
minimal \
scipy \
datascience \
mll


gpu-build:
	@python3 replace_container.py $(SOURCE_PATH) $(NEW_BASE)
	@cd docker-stacks && make build-all OWNER=samuel62


tag-all:
	@docker tag $(OWNER)/base-notebook:latest $(OWNER)/machine_learning_lab:base_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/minimal-notebook:latest $(OWNER)/machine_learning_lab:minimal_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/scipy-notebook:latest $(OWNER)/machine_learning_lab:scipy_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/datascience-notebook:latest $(OWNER)/machine_learning_lab:datascience_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/all-spark-notebook:latest $(OWNER)/machine_learning_lab:all_spark_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/pyspark-notebook:latest $(OWNER)/machine_learning_lab:pyspark_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/r-notebook:latest $(OWNER)/machine_learning_lab:r_cuda_$(CUDA_VER)
	@docker tag $(OWNER)/tensorflow-notebook:latest $(OWNER)/machine_learning_lab:tensorflow_cuda_$(CUDA_VER)

push-all:
	@docker push $(OWNER)/machine_learning_lab:base_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:minimal_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:scipy_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:datascience_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:all_spark_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:pyspark_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:r_cuda_$(CUDA_VER)
	@docker push $(OWNER)/machine_learning_lab:tensorflow_cuda_$(CUDA_VER)


install-dependencies:
	@pip3 install -r requirements.txt


test/%: ## run tests for each image
	@python3 tests/exec_tests.py $(notdir $@) $(OWNER) $(CUDA_VER)
test-all: $(foreach I, $(ALL_LIST), test/$(I)) ## test all docker-stack images


docs/%: ## generate documentation for each image
	@python3 utils/generate_docs.py $(OWNER)/machine_learning_lab:$(notdir $@)_cuda_$(CUDA_VER)
docs-all: $(foreach I, $(LAB_LIST), docs/$(I)) ## generate all docs


format:
	@black --verbose --exclude=docker-stacks .