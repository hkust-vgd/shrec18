# SHREC 2017: RGB-D Object-to-CAD Retrieval

This repository contains detailed description of the dataset and supplemental
code for [SHREC 2018: RGB-D Object-to-CAD Retrieval](http://people.sutd.edu.sg/~saikit/projects/sceneNN/shrec18/index.html).
In this track, our goal is to retrieve a CAD model from ShapeNet using a SceneNN
model as input. More details can be found at our main website.

# Evaluation
We provide Python evaluation scripts for all of the metrics and the ground truth
categories. All bug reports and suggestion are welcomed.

Usage:

```python evaluate.py <path_to_retrieval_results>```

## Acknowledgement
Some RGB-D objects in this dataset are extracted from
[ScanNet](http://www.scan-net.org/), a richly annotated 3D reconstructions of
indoor scenes.

The CAD models in this dataset are extracted from
[ShapeNet](https://www.shapenet.org/), a richly annotated and large-scale
dataset of 3D shapes by Stanford.
