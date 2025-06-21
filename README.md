# Object Pose Estimation Solution for the 1st Perception Challenge for Bin-Picking
_A Solution submission leveraging [YOLO11](https://docs.ultralytics.com/models/yolo11/) and [FoundationPose](https://nvlabs.github.io/FoundationPose/), with custom refinements_

## About This Project
This repository contains the main codebase and documentation for my entry in the 1st edition of the _Perception Challenge for Bin-picking_ (the "BPC Challenge"), an international computer vision competition sponsored by Intrinsic and hosted by OpenCV. The challenge focuses on advancing robust 6DoF object pose estimation solutions for challenging industrial parts in bin-picking scenarios.

**What is 6DoF Object Pose Estimation?**
6DoF (6 Degrees of Freedom) object pose estimation is a computer vision task that determines both the 3D position and 3D orientation of objects in space. The "6 degrees" refer to the three translational movements (X, Y, Z coordinates) and three rotational movements (roll, pitch, yaw angles) that completely describe an object's pose in 3D space. This technique is crucial for robotics applications like bin-picking, where robots need to precisely locate and grasp objects that may be randomly oriented and partially occluded in cluttered environments.

More information about the challenge can be found on the [official website](https://bpc.opencv.org/)—there, click on the _Participate_ button to access more details.

### Acknowledgements

I would like to extend special thanks to **Nuno Salgueiro** and **Wolf Byttner**—both accomplished computer vision engineers and long-time friends—for their valuable insights and feedback throughout this project. Although the vast majority of the work and implementation is my own (as they were busy with other commitments), their discussions and suggestions were instrumental in refining key ideas. Together, we formed the team _Binsightful Pose_.

### Challenge Structure

The BPC Challenge was organized into three distinct phases, each aimed at assessing different facets of object pose estimation.

- **Phase 1: Development**  
  In this phase, teams were tasked with developing pose estimation solutions for 10 designated test parts. To support this, the organizers provided two types of datasets: a large synthetic dataset (intended for model development and training) and a real image dataset (intended for validation and for evaluating solution submissions). Both datasets, along with 3D models of the parts, were made available to all teams. Each team was permitted up to 50 solution submissions.

- **Phase 2: Testing**
  This phase evaluated the ability of solutions to generalize to entirely new, unseen objects. Teams were presented with 10 brand-new parts and were limited to just 4 submission attempts. Only 3D models and synthetic training data were provided—no real images—making it essential to avoid overfitting to the development set. The organizers noted that these 10 new parts were more challenging than the first set, with some featuring very similar shapes and others being nearly symmetrical, which could easily confuse both object detection and pose estimation algorithms.

  **Phase 3: Robot-in-the-loop Testing**
  This phase evaluated real-world performance by assessing successful detections, picks, and placements of physical parts, while using a real robotic arm. This phase used the exact same parts and team submissions made for Phase 2 without further modifications.

### Competition Results

You can view the official public leaderboard for the first two phases of the BPC Challenge [here](https://bpc.opencv.org/web/challenges/challenge-page/1/leaderboard/1), where my team, **Binsightful Pose**, is listed among the participants. I achieved a strong 6th place finish in the first phase, which I consider a solid result given the high level of competition from both academic and industry research teams, while I finished 14th in the second phase.

### Technical Challenges: The Sim2Real Gap

Qualitatively, my solution demonstrated strong and reliable performance on the synthetic datasets in both phases. However, there was a noticeable drop in accuracy when evaluated on the real, hidden dataset in the second phase. While it is difficult to draw definitive conclusions without further analysis—mainly due to time constraints—this discrepancy likely underscores the well-known "sim2real gap." This is a common challenge in computer vision and other fields, where algorithms where algorithms that work well on simulated (synthetic) data often struggle to achieve similar results on real-world data due to differences in appearance, noise, and other factors.

### Reflections & Impact

While I did not reach the top positions in the second phase, I am grateful for the opportunity to participate and have my work evaluated alongside many impressive research teams. I worked mostly independently on this project, and the experience was both challenging and rewarding. Most importantly, it was an immense learning opportunity that helped me grow my skills in object pose estimation and practical computer vision. I hope that sharing my approach and code here will be helpful to others interested in this area.

## Solution Architecture

The solution follows a straightforward pipeline for 6D pose estimation:

[Input Image] → [YOLO11 Detection] → [Detection Filtering] → [FoundationPose Estimation] → [6D Pose Output]

The solution combines multiple deep learning models with custom optimizations:

- **YOLO11**: Initial 2D object detection using a unified multi-class model
- **FoundationPose**: 6D object pose estimation using YOLO11 detections as input
- **Custom Refinements**: 
  - Detection filtering to remove YOLO false-positives using hard-coded rules based on object types/classes and geometric constraints
  - Dynamic model loading/unloading to optimize GPU memory usage and cache clearing at strategic points during inference runtime.
- **Alternative Approach**: SAM6D with FastSAM for end-to-end detection and pose estimation (available in `sam6d-exploration` branch, not currently maintained)

This project is built on top of the official baseline solution provided by the competition organizers, which uses Docker and ROS2 to speed up development and facilitate testing.

If you are new to this challenge or to this repository, it is strongly recommended to first review the [`baseline_solution` branch of the opencv/bpc](https://github.com/opencv/bpc/tree/baseline_solution) repository. Familiarizing yourself with its structure, workflow, and evaluation pipeline will give you important context for understanding the customizations and improvements made in this codebase.


## Repository Structure

```
bpc-challenge/
├── opencv/                    # Git Subtree
│   └── bpc/
│       ├── Dockerfile.estimator
│       ├── Dockerfile.tester
│       ├── LICENSE
│       ├── README.md
│       └── bpc_baseline/      # Git Submodule
├── FoundationPose/            # Git Submodule
├── .gitignore
├── .gitmodules
├── .gittrees
├── README.md                  # This README file
└── ros2-jazzy-jalisco.dockerfile
```

### External Integrations

**Git Subtree: opencv/bpc**
- References the [opencv/bpc](https://github.com/opencv/bpc.git) repository (`baseline_solution` branch), provided by the competition organizers for bootstrapping participants' solutions.
- Contains the main estimation, testing, and YOLO11 inference scripts
- Includes significant custom modifications (see Custom Modifications section)
- Tracked in `.gittrees` file for manual subtree management (sadly Git does not seamlessly manage Subtrees as it does for Submodules...)

**Git Submodules:**
- **FoundationPose/**: Tracks [joao-gueifao-924/FoundationPose](https://github.com/joao-gueifao-924/FoundationPose.git) - core pose estimation algorithm
- **opencv/bpc/bpc_baseline/**: Tracks [joao-gueifao-924/bpc_baseline](https://github.com/joao-gueifao-924/bpc_baseline) (`single-yolo-model-grey-plus-depth` branch) - YOLO11 2D detection

## Custom Modifications

### opencv/bpc Subtree Changes
- **Unified YOLO Model**: Replaced the original one-model-per-object-class approach (which used multiple pre-trained YOLO11n instances provided by competition organizers) with a single multi-class YOLO11m model. This new model was trained specifically on the 10 parts required for phase 2.
- **FoundationPose Integration**: Main inference script delegates to FoundationPose for pose estimation
- **Detection Filtering**: Hard-rules filtering based on object types/classes and spatial constraints
- **Enhanced Dockerfiles**: Improved estimator and tester Docker environments
- **Debugging Support**: Added `debugpy` for remote debugging in Docker

### FoundationPose Optimizations
All custom optimizations for FoundationPos are documented directly in the [joao-gueifao-924/FoundationPose](https://github.com/joao-gueifao-924/FoundationPose.git) repository. For details on the specific improvements and techniques applied, please refer to that repository's documentation and commit history.

Key optimizations include:
- **Memory optimization**: Reduced GPU memory usage to accommodate development on a laptop with RTX 4060 (8GB VRAM)
- **Inference Speedup**: Optimized PyTorch runtime configuration to accelerate inference - reducing total processing time from nearly 12 hours to approximately 5.5 hours.
- **Dependency modernization**: Streamlined the software dependency graph and updated to the latest versions, particularly CUDA and PyTorch libraries

As a side note, similar video memory optimizations were also made for SAM-6D. Refer to `sam6d-exploration` branch commit history if interested to know more.

## Development Status
- **Main Development Branch**: [`foundation-pose-phase2-submission`](https://github.com/joao-gueifao-924/bpc-challenge/tree/foundation-pose-phase2-submission) (pending merge into `main`)
- All recent enhancements and subtree/submodule updates are on this branch

## How to Use

**Important**: Before proceeding, please familiarize yourself with the README for the [`baseline_solution` branch of the opencv/bpc](https://github.com/opencv/bpc/tree/baseline_solution) repository, particularly the [Setting up](https://github.com/opencv/bpc/tree/baseline_solution?tab=readme-ov-file#setting-up-) section. The instructions below are adapted from that documentation.


#### Setup a workspace
```bash
mkdir -p ~/bpc_ws
```

#### Create a virtual environment 

If you're already working in some form of virtualenv you can continue to use that and install `bpc` in that instead of making a new one. 

```bash
python3 -m venv ~/bpc_ws/bpc_env
```

#### Activate that virtual env

```bash
source ~/bpc_ws/bpc_env/bin/activate
```

For any new shell interacting with the `bpc` command you will have to rerun this source command.

#### Install bpc 

Install the bpc command from the ibpc pypi package. (bpc was already taken :-( )

```bash
pip install ibpc
```

#### Fetch the source repository with submodules

```bash
cd ~/bpc_ws
git clone --recurse-submodules https://github.com/joao-gueifao-924/bpc-challenge.git
cd bpc-challenge
git checkout foundation-pose-phase2-submission
```

#### Fetch the dataset

```bash
cd ~/bpc_ws/bpc
bpc fetch ipd
```
This will download the ipd_base.zip, ipd_models.zip, and ipd_val.zip (approximately 6GB combined). The dataset is also available for manual download on [Hugging Face](https://huggingface.co/datasets/bop-benchmark/ipd).



#### Build the Docker images, in the following order:
```bash
# Build FoundationPose base image first:
docker build -t foundationpose_image -f FoundationPose/docker/dockerfile .

# Build ROS2 intermediary image, which bases upon FoundationPose image:
docker build -t ros2_jazzy_image -f ros2-jazzy-jalisco.dockerfile .

# Build the final estimator image, which bases upon ROS2 image:
docker build -t bpc_estimator_image -f opencv/bpc/Dockerfile.estimator .
```

#### Run the evaluation pipeline:
```bash
bpc test bpc_pose_estimator:example ipd
````

## License
See `opencv/bpc/LICENSE` and respective licenses of submodules/subtrees for usage terms.
