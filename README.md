# Node Classification with Graph Neural Networks
This project focuses on node classification in graph-structured data. We implemented multiple graph neural network models, including GCN, GAT, GraphSAGE, MPNN, and DGI. We applied preprocessing steps such as feature normalization and adjacency matrix regularization. To improve performance, we adopted an ensemble approach that combines model predictions. The final model achieved a test accuracy of 83.47% (the best one), which demonstrated the effectiveness of combining multiple graph neural network architectures with ensemble learning.

*This is a course project for CSE 881 Data Mining at MSU. This was a competition among teams where we were graded based on model accuracy. Test labels were not provided.

## Table of Contents
1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Modeling Rationale](#modeling-rationale)
4. [Modeling Implementation](#modeling-implementation)
5. [Training Process](#training-process)
6. [Ensemble Strategy](#ensemble-strategy)

## Dataset
There are 2480 nodes in total and each node belongs to one of 7 classes. Each node in the graph is associated with a feature vector and a label (class). The dataset contains 10100 edges, 1390 features, and 496/1984 train/test nodes.

The provided files include:

1. An adjacency matrix stored in `adj.npz`: Each entry in this matrix indicates whether two nodes are connected or not. E.g., adj[i, j] = 0 if node i and node j are disconnected, adj[i, j] > 0 otherwise.
   
2. A feature matrix stored in `features.npy`: Each row represents the feature vector of a node. E.g., features[i] represents the feature vector of node i.
   
3. A list of labels stored in `labels.npy`: Includes class labels of training labels.
   
4. Data splits (train/test splits) stored in `splits.json`: splits[‘idx_train’]: node index for training nodes. splits[‘idx_test’]: node index for testing nodes.

## Data Preprocessing
This project used the PyTorch geometric framework due to its efficiency in handling graph-structured data. Our implementation process followed:

• Loading the adjacency matrix, feature matrix, train/test splits and labels.

• Converting the adjacency matrix into a dense array.

• A full label array was then constructed, setting labels for training nodes, while marking test nodes with a -1.

• Feature preprocessing was used to normalize the node features, enhancing training stability. The function performs row-wise normalization of the feature matrix, ensuring each node’s feature vector equals 1. This helps prevent numerical instabilities.

• For the adjacency matrix, the standard normalization technique was applied. The function adds self-loops to the adjacent matrix by adding the identity matrix, computes the normalized adjacency matrix, and replaces infinite values with 0.

• After cleaning the data, the data was converted to PyTorch tensors and PyTorch Geometric Data objects.

## Modeling Rationale
**Graph Convolutional Network**

• This was implemented as the baseline approach, based on previous proven effectiveness on node classification tasks. Our GCN model collects features from each node’s neighbourhood through a series of graph convolutions. For a given node, the model combines the node’s features with the features of its neighbour, which allows for information to spread within the graph structure.

• GCN captures structural relationships without overfitting to noise. The added dropout helped balance model capacity and generalization.

**Graph Attention Network(GAT)**

• To improve on GCN’s equal weighting of all neighbours, GAT was implemented. GAT uses attention mechanisms, which allows the model to assign different importance to different neighbors, based on their features. The attention mechanism is trained to focus on the most relevant neighbours for classification. The GAT model implements multi-head attention to stabilize the learning process.

• GAT provided flexibility by allowing the model to focus on important neighbours. Also, the multi-head attention mechanism can stabilize the learning process and achieve a diversified aggregation of neighbour nodes. It is especially suitable for nodes with heterogeneous neighbourhoods.

**GraphSAGE (Graph Sample and Aggregation)**

• For better scalability of the nodes, GraphSAGE was used. GraphSAGE uses a sampling-based approach; instead of using all the neighbours, GraphSAGE samples a fixed number
of neighbours and combines their features.

• Sampling neighbours reduced memory and computational cost. The mean aggregator offered a simple yet powerful summarization, and GraphSAGE proved valuable in handling the sparse adjacency structure without losing crucial information.

**Message Passing Neural Network(MPNN)**

• MPNN allows for custom message-passing operations between nodes. We used two message-passing iterations to capture deeper dependencies.

• The implementation of MPNN included:

– A message function that computed messages between connected nodes.

– An update function that updated node states based on received messages.

– A readout function that generates final node representations.

• MPNN supports flexible updating and propagation of experimental node features. Although the computation is large, it improves robustness when node relationships
are complex.

**Deep Graph Infomax(DGI) and DeepInfoMaxModel**

• An infomax model learns node representations by maximizing similar information between local and global graph structures. This was used to leverage the unlabeled portions of the graph. We use GCN as the encoder and add an additional classifier. The approach initialized node implementation before tuning with unsupervised learning on the labelled training data. Finally, our model is able to understand the graph structure more comprehensively.

## Modeling Implementation
**GCN**

– Three GCN layers with 256 hidden units.

– ReLU activation between layers.

– Dropout 0.5 for regularization.

– Final softmax activation for class probabilities.

**GAT**

– Multi-head attention with 8 heads in the first layer.

– There is a hidden dimension of 32 per attention head(256 total).

– The second GAT layer has 256 units.

– ELU activation occurs between layers.

– Dropout 0.6 for regularization.

**GraphSAGE**

– The number of neighbours involved is determined according to SAGEConv from PyTorch Geometric. By default, we use all available neighbors.

– Uses the mean aggregator function (default).

– Has hidden dimensions of 256.

– Dropout 0.5 for regularization.

– ReLU applied between layers.

**MPNN**

– Uses custom message and update functions.

– There are two message-passing iterations.

– The hidden dimensions match the other models.

– Used mean aggregation to summarize neighbourhood information.

– Dropout 0.5 for regularization.

– ReLU applied between layers.

**DGI**

– The encoder has the same architecture as GCN.

– The discriminator is used to distinguish real vs corrupted samples.

## Training Process

• Models were trained using train idx, validated on val idx, and tested on idx test.

• Early stopping was implemented with a patience of 20 epochs to prevent overfitting.

• For the loss function, we use negative loglikelihood loss. This method is more stable, it can prevent numerical overflow, and punish wrong classifications, which can help the model learn better. The performance plots for each model can be found in the code file.

## Ensemble Strategy
To improve the classification performance, we adopted an ensemble learning method. Each trained model provided predictions, weighted by the validation accuracy. The final classification was found through a weighted ensemble of predictions from every model. 

The ensemble model made up for the shortcomings of each individual model. In the end, we achieved a test accuracy of 83.47% (the best result). Since different models provide complementary insights, we thereby improve the overall performance.
