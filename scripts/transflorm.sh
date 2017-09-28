/home/evgeny/lib/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=$1 --out_graph=$2 \
--inputs='input_x' \
--outputs='output' \
--transforms='add_default_attributes strip_unused_nodes(type=float, shape="1,256,256,3") remove_nodes(op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms quantize_weights quantize_nodes strip_unused_nodes sort_by_execution_order'