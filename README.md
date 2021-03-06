# Extract Inception weights

Simple extraction of specific layer weights (embeddings) from the pre-trained Inception net.

#### Requirements:
Tested and working on:

 * [TensorFlow 0.12.0-rc1](https://github.com/tensorflow/tensorflow/releases/tag/0.12.0-rc1)
 * [TensorFlow 1.1.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.1.0)

It's so simple that it should work on pretty much any version.

#### How to use:

1. Put all you images into 'img' folder
2. python extract_embeddings.py --layer="pool_3:0"

All the images must be in JPEG format.

#### Optional arguments:

 * --model_dir = Path to classify_image_graph_def.pb. Default is 'model_dir'. 
              If it doesn't exist it will be created and Inception net will be automatically downloaded to it.           
 * --image_dir = Path to directory containing images. Default is 'img'. 
 * --embed_dir = Path to embedding dir - another directory will be created inside, named after the chosen layer. Default is 'embeddings'.
 * --layer = Name of the hidden layer to extract weights from. Default is 'pool_3:0' which is the next to last layer.
 * --visualize_graph = True/False whether you want to visualize the graph in Tensorboard.