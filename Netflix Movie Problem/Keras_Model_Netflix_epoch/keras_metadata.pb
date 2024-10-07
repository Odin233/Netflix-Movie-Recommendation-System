
�4root"_tf_keras_network*�3{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Item"}, "name": "Item", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "User"}, "name": "User", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "Movie-Embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 17770, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "Movie-Embedding", "inbound_nodes": [[["Item", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "User-Embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 480189, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "User-Embedding", "inbound_nodes": [[["User", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "FlattenMovies", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "FlattenMovies", "inbound_nodes": [[["Movie-Embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "FlattenUsers", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "FlattenUsers", "inbound_nodes": [[["User-Embedding", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "name": "DotProduct", "inbound_nodes": [[["FlattenMovies", 0, 0, {}], ["FlattenUsers", 0, 0, {}]]]}], "input_layers": [["User", 0, 0], ["Item", 0, 0]], "output_layers": [["DotProduct", 0, 0]]}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "User"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "Item"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Item"}, "name": "Item", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "User"}, "name": "User", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "Movie-Embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 17770, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "Movie-Embedding", "inbound_nodes": [[["Item", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Embedding", "config": {"name": "User-Embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 480189, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "User-Embedding", "inbound_nodes": [[["User", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "FlattenMovies", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "FlattenMovies", "inbound_nodes": [[["Movie-Embedding", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Flatten", "config": {"name": "FlattenUsers", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "FlattenUsers", "inbound_nodes": [[["User-Embedding", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "name": "DotProduct", "inbound_nodes": [[["FlattenMovies", 0, 0, {}], ["FlattenUsers", 0, 0, {}]]], "shared_object_id": 8}], "input_layers": [["User", 0, 0], ["Item", 0, 0]], "output_layers": [["DotProduct", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 12}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 13}, {"class_name": "RootMeanSquaredError", "config": {"name": "rmse", "dtype": "float32"}, "shared_object_id": 14}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "Item", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Item"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "User", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "User"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "Movie-Embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "Movie-Embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 17770, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["Item", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "User-Embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "User-Embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 480189, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["User", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�root.layer-4"_tf_keras_layer*�{"name": "FlattenMovies", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "FlattenMovies", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Movie-Embedding", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 15}}2
�root.layer-5"_tf_keras_layer*�{"name": "FlattenUsers", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "FlattenUsers", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["User-Embedding", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 16}}2
�root.layer-6"_tf_keras_layer*�{"name": "DotProduct", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "inbound_nodes": [[["FlattenMovies", 0, 0, {}], ["FlattenUsers", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20]}, {"class_name": "TensorShape", "items": [null, 20]}]}2
�Iroot.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 17}2
�Jroot.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 12}2
�Kroot.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 13}2
�Lroot.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "RootMeanSquaredError", "name": "rmse", "dtype": "float32", "config": {"name": "rmse", "dtype": "float32"}, "shared_object_id": 14}2