.PHONY: install test lint format run clean docker-build logo-train logo-eval logo-predict logo-augment

install:
	poetry install

test:
	poetry run pytest tests/ -v

lint:
	poetry run flake8 src/ tests/

format:
	poetry run black src/ tests/

logo-train:
	poetry run python -m paint_box.logo_detection.train

logo-eval:
	poetry run python -m paint_box.logo_detection.cli eval --test-dir data/logo/test

logo-predict:
	poetry run python -m paint_box.logo_detection.cli predict

logo-augment:
	poetry run python -m paint_box.logo_detection.cli augment

logo-quick-eval:
	poetry run python -m paint_box.logo_detection.cli quick-eval

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ dist/ build/
	rm -rf .venv venv/

docker-build:
	docker build -t paint-box .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data paint-box
