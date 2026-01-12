# RadPlanet-Vision: Problem Formulation

## 1. Scientific Problem
**Problem**: Automated Martian Terrain Classification for Rover Path Planning and Scientific Analysis.

**Objective**: To develop a Deep Learning model capable of classifying Martian terrain types (e.g., sandy, rocky, bedrock, loose soil) using monocular imagery from Mars Rovers (Curiosity/Perseverance).

**Justification for AI**: 
- **Latency**: Manual classification by Earth-based operators has high latency (4-24 mins one way). On-board AI allows for real-time decision making.
- **Volume**: The volume of data returned by rovers is massive; automated indexing and analysis helps in prioritizing data downlink.
- **Consistency**: AI models provide consistent classification criteria compared to potentially subjective human analysis.

## 2. Assumptions & Constraints
- **Data Source**: RGB and Grayscale images from NASA's Mars Rover Photos API.
- **Labeling**: Since we lack ground-truth labelled masks for every image, we will categorize whole images or patches based on dominant terrain features, or use unsupervised/semi-supervised techniques if labels are scarce. *Initial Approach*: Weakly supervised classification using image metadata or visual clustering, or manually building a small labeled dataset from the fetched images.
- **Environment**: Varying lighting conditions, dust storms, and camera artifacts are expected noise sources.

## 3. Success Metrics
- **Accuracy**: Top-1 and Top-3 classification accuracy on a held-out test set.
- **Explainability**: Saliency maps (Grad-CAM) to verify the model is looking at the ground textures and not the sky/rover parts.
- **Efficiency**: Model inference time suitable for potential edge deployment (though running on desktop for this simulation).
