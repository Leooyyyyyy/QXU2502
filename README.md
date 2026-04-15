QXU 2502

This project focuses on developing a real-time posture correction system for yoga practice,
assisting solo practitioners in identifying and correcting improper postures. In the first
semester, the system employs a two-stage pipeline: the first stage uses a pre-trained
human pose detection model to extract body keypoints from input images, while the
second stage applies a deep learning classification model to classify yoga poses and
evaluate their correctness.

A comprehensive performance benchmark on four mainstream human pose detection
models was conducted, includin OpenPose, PoseNet, MoveNet, and BlazePose, offering a
more thorough evaluation than existing studies. Based on our evaluation, BlazePose was
selected as the model for the system's first stage. To address the lack of suitable datasets,
we constructed a custom dataset of over 3,800 yoga posture images, including both
correct and incorrect samples for four poses (Down Dog, Plank, Side Plank, and Warrior II).
For the system's second stage, we designed and trained a Shared MLP-based
classification model, which demonstrated faster convergence and better generalization
compared to a standard MLP.

In the second semester, the project was extended from static image-based posture
correction to a finer-grained real-time correction system. The dataset was refined by
reorganizing negative samples into detailed sub-posture categories, and the classification
pipeline was reformulated into a three-level design for posture type, posture correctness,
and negative sub-posture prediction.

After phased experiments and comparative evaluation, the final selected model was
integrated into a live webcam pipeline. Runtime optimization was further introduced
through latest-frame processing, temporal smoothing, and conservative sub-posture
display gating. A simple 