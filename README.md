# Node Classification with Graph Neural Networks
This classification problem aims to predict the class label for each node based on their features and the relationships they have with other nodes.

*This is a course project for CSE 881 Data Mining at MSU. Since it is a competition among teams and we will be evaluated (and graded) based on accuracy, the code cannot be shared until the final grading has been concluded. 

## Table of Contents:
1. [Introduction](#introduction)
2. [Datasets](#datasets)

## Introduction:
In modern data mining tasks, graphs are a powerful representation of relationships. For example, in a social network (e.g., Facebook, LinkedIn), the interaction of users can form a graph, where each node represents one user and edges represent social interactions, such as friendships, follows, or interactions on social media platforms. Node classification is a key task that predicts labels or categories for nodes within a graph. Thus, this project will focus on the task of node classification in a single graph. 

## Datasets:
There are 2480 nodes in total and each node belongs to one of 7 classes. Each node in the graph is associated with a feature vector and a label (class). The connection between nodes is stored in an adjacent matrix.

The provided files include:
1. An adjacency matrix: stored in file adj.npz
   Each entry in this matrix indicates whether two nodes are connected or not
   E.g., adj[i, j] = 0 if node i and node j are disconnected, adj[i, j] > 0 otherwise
2. A feature matrix: stored in file features.npy
   Each row represents the feature vector of a node
   E.g., features[i] represents the feature vector of node i
3. A list of labels: stored in file labels.npy
   Include class labels of training labels
4. Data splits (train/test splits): stored in splits.json
   splits[‘idx_train’]: node index for training nodes
   splits[‘idx_test’]: node index for testing nodes
