
# 6DoF Pose Estimation Solution for the 1st Perception Challenge for Bin-Picking
_A Solution submission leveraging YOLOv11 and FoundationPose, with custom refinements_

## Note
This README is currently under active development. Some sections may be incomplete or inconsistent as updates are ongoing. Thank you for your understanding as I work to enhance the clarity and quality of this documentation.

## About This Project
This repository contains the codebase developed for my participation in the 1st edition of the _Perception Challenge for Bin-picking_ (the "BPC Challenge"). It specifically documents my work on this computer vision competition.

## Overview
The BPC Challenge, sponsored by Intrinsic and hosted by OpenCV, aims to advance robust 6DoF pose estimation solutions for some of the most challenging industrial parts. This repository provides a detailed account of my approach, implementation, and results for the challenge.

Currently, the effective main development branch is [`foundation-pose-phase2-submission`](https://github.com/joao-gueifao-924/bpc-challenge/tree/foundation-pose-phase2-submission), which is pending merge into `main`. 

The project leverages several advanced components, including custom modifications to third-party code and integration with external repositories via both Git submodules and subtrees:
- YOLO 11 for 2D object detection
- FoundationPose for 6D object pose estimation
- SAM6D as an alternative for end-to-end object detection and pose estimation (not currently maintained)
- Memory footprint reduction by loading/unloading deep learning models between host and GPU RAM, and obliterating memory caches at specific inference runtime points

## Repository Structure

```
bpc-challenge/
├── opencv/
│   └── bpc/
│       ├── Dockerfile.estimator
│       ├── Dockerfile.tester
│       ├── LICENSE
│       ├── README.md
│       └── bpc_baseline/      # Submodule
├── FoundationPose/            # Submodule
├── .gitignore
├── .gitmodules
├── .gittrees
├── README.md
└── ros2-jazzy-jalisco.dockerfile
```

- **opencv/bpc/**: Contains the main estimator and testing scripts, Dockerfiles, and a subtree to a third-party repository with custom modifications.
- **FoundationPose/**: Integrated as a submodule, this provides the core pose estimation functionality.
- **bpc_baseline/**: A submodule within `opencv/bpc`, tracking a specific branch for baseline solutions.


## External Integrations

### Git Subtree: opencv/bpc

- The `opencv/bpc` directory is managed as a Git subtree, referencing the [opencv/bpc](https://github.com/opencv/bpc.git) repository, specifically the `baseline_solution` branch.
- This subtree is tracked in the `.gittrees` file, which is manually maintained to document subtree updates.
- Significant modifications have been made to the subtree code, including:
    - Integration of a new unified (multi-class) YOLO model and detection checks.
    - Addition of ROI cropping around YOLO detections for improved pose estimation.
    - Filtering detections by object IDs and background clutter.
    - Visual debugging support (3D bounding box rendering).
    - GPU memory detection and dynamic image scaling for VRAM optimization.
    - Enhanced Dockerfiles for estimator and tester environments.
    - Debugging support via `debugpy` for remote debugging in Docker.


### Git Submodules

- **FoundationPose/**: Tracks [joao-gueifao-924/FoundationPose](https://github.com/joao-gueifao-924/FoundationPose.git). Regularly updated to the latest commit, providing the core pose estimation algorithm.
- **opencv/bpc/bpc_baseline/**: Tracks [joao-gueifao-924/bpc_baseline](https://github.com/joao-gueifao-924/bpc_baseline) on the `single-yolo-model-grey-plus-depth` branch, ensuring compatibility with the latest detection and pose estimation pipelines.


## Commit History Highlights

### opencv/bpc (Subtree)

Recent commit activity demonstrates an iterative and feature-driven development approach, including:

- **YOLO Model Integration**: Multiple commits updating the YOLO model usage, including switching to a single model and refining detection filtering.
- **Detection Filtering**: Added options to filter detections by object IDs and spatial constraints (e.g., x_range for background clutter).
- **ROI Cropping**: Enhanced pose estimation by cropping around detected objects.
- **Visual Debugging**: Added 3D bounding box rendering for easier debugging.
- **Performance Optimizations**: Dynamic image scaling based on GPU memory and Dockerfile improvements for CUDA/ROS2 integration.
- **Submodule Updates**: Frequent updates to the `bpc_baseline` submodule, ensuring alignment with upstream changes and new features.


### FoundationPose (Submodule)

- The `FoundationPose` submodule has been regularly updated to the latest commit, ensuring the estimator benefits from upstream improvements and bug fixes.
- No custom code changes are tracked within this repository for `FoundationPose`; updates are managed via submodule pointers. Please refer to [joao-gueifao-924/FoundationPose](https://github.com/joao-gueifao-924/FoundationPose.git) for further details and optimizations that were made.


## Development Branch

- The primary development occurs on the `foundation-pose-phase2-submission` branch, which includes all recent enhancements and subtree/submodule updates. This branch is pending merge into `main`.


## How to Use

**Clone with submodules:**

```bash
git clone --recurse-submodules https://github.com/joao-gueifao-924/bpc-challenge.git
cd bpc-challenge
git checkout foundation-pose-phase2-submission
```

**Docker-based workflows:**
The build is done in three steps, one per Docker image. Each docker image depends on the previous one:
1. Build FoundationPose Docker image
2. Build ROS2 Jazzy Docker image
3. Build BPC Challenge Docker image - use the provided `Dockerfile.estimator` and `Dockerfile.tester` in `opencv/bpc/` for reproducible builds and testing environments.


## Custom Modifications

- The subtree in `opencv/bpc` is not a vanilla copy; it contains significant custom logic for detection filtering, pose estimation, and debugging.
- FoundationPose submodule itself contains several optimizations that were made.
- All subtree updates and their rationale are documented in the `.gittrees` file.


## License

- See `opencv/bpc/LICENSE` and the respective licenses of submodules/subtrees for usage terms.
