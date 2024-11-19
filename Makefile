# Export env variables from .env file if it exists.
ifneq (,$(wildcard .env))
include .env
export
endif

### Install ###

install:
	@echo "Installing streaming pipeline..."
	
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring pip install -r requirements.txt

### Run ###

run_real_time:
	RUST_BACKTRACE=full python -m bytewax.run ingestion_pipeline:build_stream_dataflow

run_batch:
	RUST_BACKTRACE=full python -m bytewax.run -p4 "ingestion_pipeline:build_batch_dataflow(last_n_days=1)"

run_mock:
	RUST_BACKTRACE=1 run python -m bytewax.run ingestion_pipeline:build_mock_dataflow


### Run Docker ###

build:
	@echo "Build docker image"

	docker build -t streaming_pipeline:latest -f deploy/Dockerfile .