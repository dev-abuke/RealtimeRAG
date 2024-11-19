#!/bin/bash
# Runs batch stream locally so make sure to install requirements.txt

# Check if the Bytewax module is installed
if ! python3.11 -c "import bytewax" &> /dev/null; then
  echo "Error: Bytewax is not installed. Please install it using 'pip install bytewax'."
  python3.11 -m pip install -r requirements.txt
  exit 1
fi

# Run the Bytewax stream
echo "Running Bytewax Batch For 1 day.."
RUST_BACKTRACE=full python3.11 -m bytewax.run ingestion_pipeline:build_batch_dataflow

if [ $? -eq 0 ]; then
  echo "Bytewax stream is running successfully."
else
  echo "Error: Failed to run Bytewax stream."
  exit 1
fi