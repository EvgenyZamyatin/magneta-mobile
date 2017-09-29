python3 -m tensorflow.python.tools.optimize_for_inference \
  --input=$1 \
  --output="$2" \
  --input_names="input_x" \
  --output_names="output"

python3 -m scripts.quantize_graph \
  --input="$2" \
  --output="$2" \
  --output_node_names="output" \
  --mode="quantize" \
  --test_input_dims="1,256,256,3"
