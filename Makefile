VERSION = $(shell cat VERSION)

# makes all commands of each target exec in the same shell
# https://www.gnu.org/software/make/manual/html_node/One-Shell.html
.ONESHELL:

.PHONY: source_code_format
source_code_format:
	black --line-length 120 --target-version py37 . && \
	isort .

.PHONY: source_code_check_format
source_code_check_format:
	black --check --line-length 120 --target-version py37 . && \
	isort --check-only . && \
	flake8 .

.PHONY: test
test:
	${MAKE} source_code_check_format || exit 1
	pytest || exit 1

.PHONY: test_ci
test_ci:
	pytest -m "not gpu" || exit 1

.PHONY: build_docker
build_docker:
	DOCKER_BUILDKIT=1 docker build \
	--rm \
	-t ghcr.io/els-rd/transformer-deploy:latest \
	-t ghcr.io/els-rd/transformer-deploy:$(VERSION) \
	-f Dockerfile .

.PHONY: build_push_docker
build_push_docker:
	! docker manifest inspect ghcr.io/els-rd/transformer-deploy:$(shell cat VERSION) > /dev/null || exit 1
	${MAKE} build_docker || exit 1
	docker push ghcr.io/els-rd/transformer-deploy:latest || exit 1
	docker push ghcr.io/els-rd/transformer-deploy:$(VERSION) || exit 1
