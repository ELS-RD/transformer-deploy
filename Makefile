# apply formating rule
.PHONY: source_code_format
source_code_format:
	black --line-length 120 --target-version py37 . && \
	isort .

# check that formating rules are respected
.PHONY: source_code_check_format
source_code_check_format:
	black --check --line-length 120 --target-version py37 . && \
	isort --check-only . && \
	flake8 .

.PHONY: test
test:
	${MAKE} source_code_check_format || exit 1
	pytest
