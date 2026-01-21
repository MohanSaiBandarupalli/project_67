.PHONY: test data

test:
	poetry run python -m pytest

data:
	poetry run python -m ntg.pipelines.build_dataset
	
.PHONY: data_duckdb

data_duckdb:
	poetry run python -m ntg.pipelines.build_dataset_duckdb
