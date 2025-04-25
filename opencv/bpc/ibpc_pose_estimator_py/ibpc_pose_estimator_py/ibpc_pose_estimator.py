# todo(Yadunund): Add copyright.
import time
import os
import cv2
import gc
from enum import Enum
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
import sys
from typing import List, Optional, Union
from types import SimpleNamespace
from bpc.inference.process_pose import PoseEstimator as PoseEstimatorBP
from bpc.inference.process_pose import PoseEstimatorParams
from bpc.utils.data_utils import Capture


from geometry_msgs.msg import Pose as PoseMsg
from ibpc_interfaces.msg import Camera as CameraMsg
from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
from ibpc_interfaces.srv import GetPoseEstimates

import rclpy
from rclpy.node import Node


import sys, torch, math

try:
    _, total_gpu_memory = torch.cuda.mem_get_info()
    total_gpu_memory /= 1024**3 # gigabytes

    # reported value will always a little less than the hardware value
    # due to Torch baseline memory footprint
    total_gpu_memory = math.ceil(total_gpu_memory)
    print(f"Total GPU memory: {total_gpu_memory:.2f} GB")
except:
    total_gpu_memory = 24 # targeting the NVIDIA L4 GPU with 24 GB of VRAM

class ALGORITHM_MODE(Enum):
    FOUNDATION_POSE = 1,
    SAM_6D          = 2,
    CIRP_BASELINE   = 3,
ALGORITHM_USED = ALGORITHM_MODE.SAM_6D

DO_DEBUG_SESSION = False
LOW_GPU_MEMORY_MODE=False
SHORTER_SIDE = 720 # pixels
OBJECT_CAD_MODEL_PATH = "/opt/ros/underlay/install/3d_models" # mounted at runtime through Docker



if DO_DEBUG_SESSION:
    import debugpy
    print("Path of the current script file: ", os.path.abspath(__file__))
    
    # Listen on all interfaces (0.0.0.0) for remote debugging, 
    # since we're in host network mode:
    debugpy.listen(("0.0.0.0", 5678))


# Import FoundationPose, located at the / root folder, for the time being
# TODO: improve this
if ALGORITHM_USED is ALGORITHM_MODE.FOUNDATION_POSE:
    sys.path.append("/")
    from estimater import PoseEstimator as FoundationPoseEstimator
    from datareader import IpdReader
    POSE_REFINER_TOTAL_ITERATIONS=5
    DEBUG_LEVEL = 1 if DO_DEBUG_SESSION else 0
    DEBUG_DIR='//debug'
    print("Instantiating PoseEstimator class...")
    foundationPoseEstimator = FoundationPoseEstimator(debug=DEBUG_LEVEL, est_refine_iter=POSE_REFINER_TOTAL_ITERATIONS, 
                                  debug_dir=DEBUG_DIR, low_gpu_mem_mode=LOW_GPU_MEMORY_MODE)
    print("Finished instantiating PoseEstimator class...")

elif ALGORITHM_USED is ALGORITHM_MODE.SAM_6D:
    SAM6D_ROOT_DIR = "/SAM-6D"
    sys.path.append(SAM6D_ROOT_DIR)
    import ipdreader, runtime_utils

    # Add the 'Instance_Segmentation_Model' directory to sys.path
    # insert(0, ...) gives it high priority, making Python look here first.
    ism_package_dir = os.path.join(SAM6D_ROOT_DIR, 'Instance_Segmentation_Model')
    if ism_package_dir not in sys.path:
        sys.path.insert(0, ism_package_dir)
    import Instance_Segmentation_Model.run_inference_custom as ISM
    import Pose_Estimation_Model.run_inference_custom as PEM
    import subprocess
    TEMPLATE_OUTPUT_ROOT_DIR = "/sam6d_obj_templates"
    BLENDER_PATH = os.getenv("BLENDER_PATH")



class running_average_calc:
    def __init__(self,):
        self.count = 0.0
        self.avg = 0.0
    def update(self, new_val: float):
        self.count += 1
        self.avg += (new_val - self.avg) / self.count


# Helper functions
def ros_pose_to_mat(pose: PoseMsg):
    r = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    matrix = r.as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = matrix
    pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return pose_matrix


class Camera:
    """
    Represents a camera with its pose, intrinsics, and image data,
    initialized from either a CameraMsg or a PhotoneoMsg ROS message.

    Attributes:
        name (str): The name of the camera (taken from the message's frame_id).
        pose (np.ndarray): The 4x4 camera pose matrix (world-to-camera transformation),
                          converted from the ROS message's pose.
        intrinsics (np.ndarray): The 3x3 camera intrinsics matrix, reshaped from the
                                  message's K matrix.
        rgb (np.ndarray): The RGB image data, converted from the ROS message.
        depth (np.ndarray): The depth image data, converted from the ROS message.
        aolp (np.ndarray, optional): The Angle of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
        dolp (np.ndarray, optional): The Degree of Linear Polarization data,
                                     converted from the CameraMsg (None for PhotoneoMsg).
    """

    def __init__(self, msg: Union[CameraMsg, PhotoneoMsg]):
        """
        Initializes a new Camera object from a ROS message.

        Args:
            msg: Either a CameraMsg or a PhotoneoMsg ROS message containing
                 camera information.

        Raises:
           TypeError: If the input `msg` is not of the expected type.
        """
        br = CvBridge()

        if not isinstance(msg, (CameraMsg, PhotoneoMsg)):
            raise TypeError("Input message must be of type CameraMsg or PhotoneoMsg")

        self.name: str = (msg.info.header.frame_id,)
        self.pose: np.ndarray = ros_pose_to_mat(msg.pose)
        self.intrinsics: np.ndarray = np.array(msg.info.k).reshape(3, 3)
        self.rgb = br.imgmsg_to_cv2(msg.rgb)
        self.depth = br.imgmsg_to_cv2(msg.depth)
        if isinstance(msg, CameraMsg):
            self.aolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.aolp)
            self.dolp: Optional[np.ndarray] = br.imgmsg_to_cv2(msg.dolp)
        else:  # PhotoneoMsg
            self.aolp: Optional[np.ndarray] = None
            self.dolp: Optional[np.ndarray] = None


def rot_to_quat(rot):
    r = Rotation.from_matrix(rot)
    q = r.as_quat()
    return q


def downscale_images(shorter_side, K, color, depth, mask=None):
    if shorter_side is not None and shorter_side > 0:
        original_shorter_side = np.min(color.shape)
        scale_factor = SHORTER_SIDE / original_shorter_side
        if (scale_factor < 1.0):
            #print("scale factor: ", scale_factor)
            K[:2] *= scale_factor
            color = cv2.resize(color, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            if mask is not None:
                mask  = cv2.resize(mask,  None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                return K, color, depth, mask
    return K, color, depth


class PoseEstimator(Node):

    def __init__(self):
        super().__init__("bpc_pose_estimator")
        self.get_logger().info("Starting bpc_pose_estimator...")
        # Declare parameters
        self.model_cache = {}
        self.object_cad_model_cache = {}
        self.model_dir = (
            self.declare_parameter("model_dir", "").get_parameter_value().string_value
        )
        if self.model_dir == "":
            raise Exception("ROS parameter model_dir not set.")
        self.get_logger().info(f"Model directory set to {self.model_dir}.")
        srv_name = "/get_pose_estimates"
        self.get_logger().info(f"Pose estimates can be queried over srv {srv_name}.")
        self.srv = self.create_service(GetPoseEstimates, srv_name, self.srv_cb)

        if ALGORITHM_USED == ALGORITHM_MODE.FOUNDATION_POSE:
            self.object_cad_model_cache, _ = IpdReader.load_object_meshes(OBJECT_CAD_MODEL_PATH)

        elif ALGORITHM_USED == ALGORITHM_MODE.SAM_6D:
            self.object_cad_model_cache, _ = ipdreader.IpdReader.load_object_meshes(OBJECT_CAD_MODEL_PATH)

            self.object_class_ids = list(self.object_cad_model_cache.keys())

            original_cwd = os.getcwd()
            target_cwd = os.path.join(SAM6D_ROOT_DIR, 'Instance_Segmentation_Model'); os.chdir(target_cwd)
            self.sam6d = SimpleNamespace()
            self.sam6d.segmentator_model, self.sam6d.device = ISM.load_model(segmentor_model="fastsam", stability_score_thresh=0.97)
            os.chdir(original_cwd)

            target_cwd = os.path.join(SAM6D_ROOT_DIR, 'Pose_Estimation_Model'); os.chdir(target_cwd)
            args = SimpleNamespace()
            args.low_gpu_memory_mode = LOW_GPU_MEMORY_MODE
            self.sam6d.pem = PEM.PoseEstimatorModel(args)
            os.chdir(original_cwd)

            if LOW_GPU_MEMORY_MODE:
                ISM.load_descriptormodel_to_gpu(self.sam6d.segmentator_model)
                self.sam6d.pem.unload_model_to_cpu()
                gc.collect()
                torch.cuda.empty_cache()

            # Get the directory where the current script is located
            #script_dir = os.path.dirname(os.path.abspath(__file__))
                
            # Define the path to the 'Render' directory relative to the script
            self.sam6d.render_dir = os.path.join("/SAM-6D", 'Render')

            # Object preparation stage. Render templates and infer descriptors out of them.
            self.sam6d.template_descriptors = {}
            for object_class_id in self.object_class_ids:
                start_time = time.time()
                mesh = self.object_cad_model_cache[object_class_id]
                runtime_utils.render_object_templates(object_class_id, OBJECT_CAD_MODEL_PATH, self.sam6d.render_dir, TEMPLATE_OUTPUT_ROOT_DIR, BLENDER_PATH)
                self.get_logger().info(f"Load 3D object mesh and rendering templates time: {time.time() - start_time} seconds")

                start_time = time.time()
                descriptors, appe_descriptors = ISM.init_templates(
                    runtime_utils.get_obj_template_dir(object_class_id, TEMPLATE_OUTPUT_ROOT_DIR), 
                    self.sam6d.segmentator_model, self.sam6d.device)
                self.sam6d.template_descriptors[object_class_id] = {}
                self.sam6d.template_descriptors[object_class_id]["descriptors"] = descriptors
                self.sam6d.template_descriptors[object_class_id]["appe_descriptors"] = appe_descriptors
                self.get_logger().info(f"infer descriptors from from templates time: {time.time() - start_time} seconds")

        self.avg_total_inference_time = running_average_calc()
        

    def srv_cb(self, request, response):
        if len(request.object_ids) == 0:
            self.get_logger().warn("Received request with empty object_ids.")
            return response
        if len(request.cameras) < 3:
            self.get_logger().warn("Received request with insufficient cameras.")
            return response
        # try:
        cam_1 = Camera(request.cameras[0])
        cam_2 = Camera(request.cameras[1])
        cam_3 = Camera(request.cameras[2])
        photoneo = Camera(request.photoneo)
        response.pose_estimates = self.get_pose_estimates(
            request.object_ids, cam_1, cam_2, cam_3, photoneo
        )
        # except:
        #     self.get_logger().error("Error calling get_pose_estimates.")
        return response

    def get_pose_estimates(
        self,
        object_ids: List[int],
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: Camera,
    ) -> List[PoseEstimateMsg]:
        pose_estimates = []

        # Use only camera 1:
        image_rgb_rows, image_rgb_cols = cam_1.rgb.shape[0], cam_1.rgb.shape[1]
        K     = cam_1.intrinsics.copy()
        color = cam_1.rgb.copy()
        depth = cam_1.depth.copy()

        # Multiply by the depth factor (0.1) to get from raw depth pixel values to millimeters,
        # then convert from lmilimeters to meters (what both FoundationPose and SAM-6D algorithms expect).
        depth *= 0.1 / 1000 # TODO put this convertion elsewhere?

        if ALGORITHM_USED == ALGORITHM_MODE.SAM_6D:
            K, color, depth = downscale_images(SHORTER_SIDE, K, color, depth)
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
            whole_pts = self.sam6d.pem.get_point_cloud_from_depth(depth, K)
            all_image_detections, query_decriptors, query_appe_descriptors = ISM.infer_on_image(color, self.sam6d.segmentator_model)
            

        for object_id in object_ids:
            start_whole_image_inference_time = time.time()
            mesh = self.object_cad_model_cache[object_id]

            if ALGORITHM_USED == ALGORITHM_MODE.FOUNDATION_POSE:
                if object_id not in self.model_cache:
                    yolo_model_path = os.path.join(self.model_dir, f'detection/obj_{object_id}/yolo11-detection-obj_{object_id}.pt')
                    pose_model_path = os.path.join(self.model_dir, f'rot_models/rot_{object_id}.pth')

                    pose_params = PoseEstimatorParams(yolo_model_path=yolo_model_path,
                                                    pose_model_path=pose_model_path, 
                                                    yolo_conf_thresh=0.1)
                    pose_estimator = PoseEstimatorBP(pose_params)
                    self.model_cache[object_id] = pose_estimator
                pose_estimator = self.model_cache[object_id]
                t = time.time()
                cams = [cam_1, cam_2, cam_3]
                images = [np.tile(x.rgb[:,:,None], (1, 1, 3)) for x in cams]
                RTs = [x.pose for x in cams]
                Ks = [x.intrinsics for x in cams]
                capture = Capture(images, Ks, RTs, object_id)
                detections = pose_estimator._detect(capture)

                # Returns a dictionary mapping camera indices to a list of detections.
                # Each detection is a dictionary with keys "bbox" and "bb_center".

                # print("list(detections.keys()): ", list(detections.keys())) # [0, 1, 2]               
                if DO_DEBUG_SESSION:
                    print(f"total detections of object #{object_id}: {len(detections[0])}")

                for detection in detections[0]:
                    start_inference_time = time.time()
                    # 'bbox': (x1, y1, x2, y2),
                    (x1, y1, x2, y2) = detection['bbox']

                    #print("BBox: ", (x1, y1, x2, y2))
                    #print("cam_1.depth: ", cam_1.depth.shape, np.sum((cam_1.depth > 0.001)),  cam_1.depth.dtype)

                    # Create a binary mask for the detected object of same size as RGB image
                    mask = np.zeros((image_rgb_rows, image_rgb_cols), dtype=np.uint8)

                    # Fill in a white rectangle using the (x1, y1, x2, y2) corner coordinates:
                    mask = cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)
                    
                    if DO_DEBUG_SESSION:
                        valid = (cam_1.depth>=0.001) & (mask>0)
                        print("Area of valid: ", valid.sum())

                    K, color, depth, mask = downscale_images(SHORTER_SIDE, K, color, depth, mask)
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
                    mask = mask.astype(bool)

                    object_pose = foundationPoseEstimator.estimate( object_class_id=object_id,
                                                                    K     = K, 
                                                                    mesh  = self.object_cad_model_cache[object_id],
                                                                    color = color,
                                                                    depth = depth,
                                                                    mask  = mask
                                                                )
                    
                    # Need to convert from meters (output by FoundationPose) to millimeters again
                    object_pose[0:3, 3] *= 1000.0

                    # Need to map from camera pose to world pose
                    # cam_1.pose is defined as transformation matrix world-to-camera, hence we need to invert it
                    # to become camera-to-world
                    object_pose = np.linalg.inv(cam_1.pose) @ object_pose

                    if DO_DEBUG_SESSION:
                        print("object_pose: ")
                        print(object_pose)
                        print("--------------------")

                    pose_estimate = PoseEstimateMsg()
                    pose_estimate.obj_id = object_id
                    pose_estimate.score = 1.0
                    
                    pose_estimate.pose.position.x = object_pose[0, 3]
                    pose_estimate.pose.position.y = object_pose[1, 3]
                    pose_estimate.pose.position.z = object_pose[2, 3]
                    rot = rot_to_quat(object_pose[:3, :3])
                    pose_estimate.pose.orientation.x = rot[0]
                    pose_estimate.pose.orientation.y = rot[1]
                    pose_estimate.pose.orientation.z = rot[2]
                    pose_estimate.pose.orientation.w = rot[3]
                    pose_estimates.append(pose_estimate)

                    inference_time = time.time() - start_inference_time
                    print(f"Foundation Pose inference time for one object instance: {inference_time:.3f} seconds")

            elif ALGORITHM_USED == ALGORITHM_MODE.SAM_6D:
                self.get_logger().info(f"Object Class ID: {object_id}")

                # 1st stage
                # Let's infer 2D object detections

                ISM.reset_ref_data(self.sam6d.segmentator_model, 
                                   self.sam6d.template_descriptors[object_id]["descriptors"],
                                   self.sam6d.template_descriptors[object_id]["appe_descriptors"]
                )

                start_time = time.time()
                obj_class_detections = ISM.run_inference(
                    model=self.sam6d.segmentator_model, 
                    device=self.sam6d.device,
                    low_gpu_mem_mode=LOW_GPU_MEMORY_MODE,
                    output_dir=None, 
                    cad_model=mesh,
                    rgb_image=color,
                    all_image_detections=all_image_detections,
                    query_decriptors=query_decriptors, 
                    query_appe_descriptors=query_appe_descriptors, 
                    depth_image=depth, 
                    cam_K=K, 
                    depth_scale=1.0,
                    min_detection_final_score=0.0
                )
                self.get_logger().info(f"ISM.run_inference time: {time.time() - start_time} seconds")


                if LOW_GPU_MEMORY_MODE:
                    ISM.unload_descriptormodel_to_cpu(self.sam6d.segmentator_model)
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.sam6d.pem.load_model_to_gpu()
                    gc.collect()
                    torch.cuda.empty_cache()

                # 2nd stage
                # Now that we have the 2D object detections, let's run pose inference on each detection.
                start_time = time.time()
                # TODO fdalkjfdkajfkj I should run these only once and cache results!!!!
                all_tem_pts, all_tem_feat = self.sam6d.pem.get_templates(
                    os.path.join(runtime_utils.get_obj_template_dir(object_id, TEMPLATE_OUTPUT_ROOT_DIR), "templates")
                )
                self.get_logger().info(f"get_templates time: {time.time() - start_time} seconds")

                start_time = time.time()
                model_points = self.sam6d.pem.sample_points_from_mesh(mesh)
                input_data, detections = self.sam6d.pem.prepare_test_data(obj_class_detections, color, depth, whole_pts, K, model_points)
                self.get_logger().info(f"sample_points_from_mesh and prepare_test_data time: {time.time() - start_time} seconds")

                start_time = time.time()
                pose_scores, pred_rot, pred_trans = self.sam6d.pem.infer_pose(all_tem_pts, all_tem_feat, input_data)
                pred_trans *= 1000.0 # convert from meters back to millimeters
                self.get_logger().info(f"infer_pose time: {time.time() - start_time} seconds")

                for i in range(len(pose_scores)):
                    object_translation = pred_trans[i].flatten()
                    object_rotation = pred_rot[i]
                    object_pose = np.eye(4)
                    object_pose[:3, :3] = object_rotation
                    object_pose[:3, 3] = object_translation

                    # Need to map from camera pose to world pose
                    # cam_1.pose is defined as transformation matrix world-to-camera, hence we need to invert it
                    # to become camera-to-world
                    object_pose = np.linalg.inv(cam_1.pose) @ object_pose

                    if DO_DEBUG_SESSION:
                        self.get_logger().info("object_pose: ")
                        self.get_logger().info(np.array2string(object_pose))
                        self.get_logger().info("--------------------")

                    pose_estimate = PoseEstimateMsg()
                    pose_estimate.obj_id = object_id
                    pose_estimate.score = pose_scores[i]
                    
                    pose_estimate.pose.position.x = object_pose[0, 3]
                    pose_estimate.pose.position.y = object_pose[1, 3]
                    pose_estimate.pose.position.z = object_pose[2, 3]
                    rot = rot_to_quat(object_pose[:3, :3])
                    pose_estimate.pose.orientation.x = rot[0]
                    pose_estimate.pose.orientation.y = rot[1]
                    pose_estimate.pose.orientation.z = rot[2]
                    pose_estimate.pose.orientation.w = rot[3]
                    pose_estimates.append(pose_estimate)

                if LOW_GPU_MEMORY_MODE:
                    self.sam6d.pem.unload_model_to_cpu()
                    gc.collect()
                    torch.cuda.empty_cache()
                    ISM.load_descriptormodel_to_gpu(self.sam6d.segmentator_model)
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            else: # this is code from baseline solution, just keeping it here for reference
                pose_predictions = pose_estimator._match(capture, detections)
                pose_estimator._estimate_rotation(pose_predictions)
                for detection in pose_predictions:
                    pose_estimate = PoseEstimateMsg()
                    pose_estimate.obj_id = object_id
                    pose_estimate.score = 1.0
                    pose_estimate.pose.position.x = detection.pose[0, 3]
                    pose_estimate.pose.position.y = detection.pose[1, 3]
                    pose_estimate.pose.position.z = detection.pose[2, 3]
                    rot = rot_to_quat(detection.pose[:3, :3])
                    pose_estimate.pose.orientation.x = rot[0]
                    pose_estimate.pose.orientation.y = rot[1]
                    pose_estimate.pose.orientation.z = rot[2]
                    pose_estimate.pose.orientation.w = rot[3]
                    pose_estimates.append(pose_estimate)

            self.avg_total_inference_time.update(time.time() - start_whole_image_inference_time)
            self.get_logger().info(f"Avg. infer. time per image: {self.avg_total_inference_time.avg} seconds")
            self.get_logger().info(f"Total inferences made: {self.avg_total_inference_time.count}")

        return pose_estimates


def main(argv=sys.argv):
    rclpy.init(args=argv)

    pose_estimator = PoseEstimator()

    rclpy.spin(pose_estimator)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
