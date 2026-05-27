# PaleoCV

PaleoCV is a modular explainable computer vision framework for AI-assisted palaeontological fossil analysis. The project combines deep learning, explainable AI, computer vision and interactive visualization to perform fossil classification, morphology-aware analysis and pseudo-3D reconstruction from fossil imagery.

The system was developed to explore how modern computer vision techniques can assist palaeontological workflows under limited-data conditions. PaleoCV integrates multiple stages of fossil analysis into a unified pipeline, including fossil classification, confidence estimation, Grad-CAM explainability, segmentation, contour extraction, morphological analysis and interactive reconstruction.

The project uses transfer learning with ResNet-based convolutional neural networks trained on the GeoFossils dataset containing multiple fossil categories such as ammonites, belemnites, corals, crinoids, leaf fossils and trilobites. Grad-CAM heatmaps are used to visualize the regions responsible for classification decisions, improving interpretability and scientific trustworthiness.

In addition to classification, PaleoCV performs morphology-aware analysis using contour-based segmentation and geometric feature extraction. Quantitative descriptors such as area, perimeter, aspect ratio, circularity and solidity are computed to support fossil shape analysis. The reconstruction module generates interactive pseudo-3D fossil meshes using Open3D and Plotly for browser-based visualization.

The final system is integrated into a Streamlit application that supports:

* fossil image upload,
* camera-based image acquisition,
* explainable fossil classification,
* segmentation,
* morphology analysis,
* and interactive reconstruction.

PaleoCV was inspired by recent research in machine learning for palaeontology and aims to provide an explainable, reproducible and extensible framework for AI-assisted fossil analysis.

