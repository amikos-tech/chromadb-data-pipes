lint:
	@echo "Running pre-commit hooks"
	@pre-commit run --all-files
pre-commit:
	@echo "Linting last commit"
	@pre-commit run --from-ref HEAD~1 --to-ref HEAD
dependencies:
	@echo "Installing dependencies"
	@poetry update
install:
	@echo "Installing the project"
	@poetry install
test:
	@echo "Running tests"
	@poetry run pytest
build-docker:
	@echo "Building docker image"
	@docker build -t chromadb-dp .
