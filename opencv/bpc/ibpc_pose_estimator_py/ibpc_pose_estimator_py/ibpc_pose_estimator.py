# todo(Yadunund): Add copyright.
import time
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
import sys
from typing import List, Optional, Union
from bpc.inference.process_pose import PoseEstimator as PoseEstimatorBP
from bpc.inference.process_pose import PoseEstimatorParams
import bpc.utils.data_utils as du
import bpc.inference.yolo_detection_filtering as ydf
from bpc.inference.yolo_detection import ObjectDetector


from geometry_msgs.msg import Pose as PoseMsg
from ibpc_interfaces.msg import Camera as CameraMsg
from ibpc_interfaces.msg import Photoneo as PhotoneoMsg
from ibpc_interfaces.msg import PoseEstimate as PoseEstimateMsg
from ibpc_interfaces.srv import GetPoseEstimates

import rclpy
from rclpy.node import Node


import trimesh
import shutil
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


DO_DEBUG_SESSION = False
USE_FOUNDATIONPOSE_ESTIMATOR = True # else use baseline solution
LOW_GPU_MEMORY_MODE=False
SHORTER_SIDE = 500 # pixels
DEPTH_IMAGE_SCALE_PX2MM = 0.1 # raw pixel value to millimeters
OBJECT_CAD_MODEL_PATH = "/opt/ros/underlay/install/3d_models" # mounted at runtime through Docker

DO_VISUAL_DEBUG = False
SYNTHETIC_DATASET_MODE = False
if DO_VISUAL_DEBUG:
    # Hijack the 3D models folder Docker volume mount to store debug output
    YOLO_DEBUG_OUTPUT_DIR = os.path.join(OBJECT_CAD_MODEL_PATH, "yolo_debug_output")
    if os.path.exists(YOLO_DEBUG_OUTPUT_DIR): 
        shutil.rmtree(YOLO_DEBUG_OUTPUT_DIR)
    os.makedirs(YOLO_DEBUG_OUTPUT_DIR)
    YOLO_DEBUG_OUTPUT_IMAGE_COUNTER = 0
    # Create a palette of colors for the bounding boxes for 10 objects
    YOLO_OBJECT_ID_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 128, 128), (128, 0, 0),
        (0, 128, 0), (0, 0, 128)
    ]


if DO_DEBUG_SESSION:
    import debugpy
    print("Path of the current script file: ", os.path.abspath(__file__))
    
    # Listen on all interfaces (0.0.0.0) for remote debugging, 
    # since we're in host network mode:
    debugpy.listen(("0.0.0.0", 5678))


# Import FoundationPose, located at the / root folder, for the time being
# TODO: improve this
if USE_FOUNDATIONPOSE_ESTIMATOR:
    sys.path.append("/")
    from Utils import draw_xyz_axis, draw_posed_3d_box
    from estimater import PoseEstimator as FoundationPoseEstimator
    from datareader import IpdReader
    POSE_REFINER_TOTAL_ITERATIONS=5
    DEBUG_LEVEL = 1 if DO_DEBUG_SESSION else 0
    DEBUG_DIR='//debug'

    # deprecated, not currently used anymore
    # def get_suitable_shorter_side(total_gpu_memory):
    #     # Use two successful experimental samples, with 6 and 8 GB VRAM GPUs
    #     x1 = 6.0 # GB
    #     y1 = 360.0 # pixels
    #     x2 = 8.0 # GB
    #     y2 = 400.0 # pixels

    #     ratio = (y2-y1) / (x2-x1) # pixels/GB, how many pixels increase per additional gigabyte of memory
    #     shorter_side = y2 + ratio * (total_gpu_memory - x2) # pixels

    #     # shorter_side will be 720 pixels for 24 GB of VRAM

    #     return shorter_side
    # shorter_side = get_suitable_shorter_side(total_gpu_memory)
    # self.printinfo("shorter_side: ", shorter_side)
        
    
    print("Instantiating PoseEstimator class...")
    foundationPoseEstimator = FoundationPoseEstimator(debug=DEBUG_LEVEL, est_refine_iter=POSE_REFINER_TOTAL_ITERATIONS, 
                                  debug_dir=DEBUG_DIR, low_gpu_mem_mode=LOW_GPU_MEMORY_MODE)
    print("Finished instantiating PoseEstimator class...")



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

class PoseEstimator(Node):

    def printinfo(self, msg):
        self.get_logger().info(msg)

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
        self.object_cad_model_cache, _ = IpdReader.load_object_meshes(OBJECT_CAD_MODEL_PATH)

        yolo_model_path = os.path.join(self.model_dir, "detection", "yolo-11-training-08-single-model-grey-plus-depth-hillshade.pt")
        yolo_thresholds_path = os.path.join(self.model_dir, "detection", "detection_confidence_thresholds.json")

        self.object_detector = ObjectDetector(yolo_model_path, yolo_thresholds_path, is_synthetic=SYNTHETIC_DATASET_MODE)
        

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

        # Uncomment this if you want the application to wait until the debugger attaches
        if DO_DEBUG_SESSION:
            self.printinfo("Waiting for debugger to attach...")
            debugpy.wait_for_client()
            self.printinfo("Debugger just attached. Continuing...")

        # Let's use only RGB camera 1 for now.
        intrinsics_K_cam_1 = cam_1.intrinsics.copy()
        image_cam_1 = cam_1.rgb.copy()
        depth_cam_1_raw_values = cam_1.depth.copy()
        depth_cam_1_metric_mm = du.transform_depth_image(depth_cam_1_raw_values, DEPTH_IMAGE_SCALE_PX2MM, max_depth_mm=5000.0)

        if DO_VISUAL_DEBUG:
            # Get RGB image from camera 1
            image_rgb_yolo_debug = image_cam_1.copy()
            # If single channel, convert to 3 channels
            if len(image_rgb_yolo_debug.shape) == 2:
                image_rgb_yolo_debug = np.tile(image_rgb_yolo_debug[:, :, None], (1, 1, 3))

        x_range = (860, 3180) # Empirical range to hide background clutter seen from camera 1 perspective. Image size is 3840x2160. This clutter is outside x_range.
        detections_all_obj_ids = self.object_detector.detect(image_cam_1, depth_cam_1_raw_values, x_range)

        excluded_elongated_object_ids=[4, 8, 9]
        detections_all_obj_ids = ydf.filter_detections(detections_all_obj_ids, depth_cam_1_metric_mm, self.object_cad_model_cache, intrinsics_K_cam_1, excluded_elongated_object_ids, meshes_are_in_millimeters=False)

        for object_id, detections_this_id in detections_all_obj_ids.items():

            if object_id not in object_ids:
                continue

            if USE_FOUNDATIONPOSE_ESTIMATOR:

                image_rgb_rows, image_rgb_cols = image_cam_1.shape[0], image_cam_1.shape[1]
                
                if DO_DEBUG_SESSION:
                    self.printinfo(f"total detections of object #{object_id}: {len(detections_this_id)}")

                for detection in detections_this_id:
                    start_inference_time = time.time()
                    # 'bbox': (x1, y1, x2, y2),
                    (x1, y1, x2, y2) = detection['bbox']
                    confidence = detection['confidence']

                    if DO_VISUAL_DEBUG:
                        # Render the bounding box on the image, its object id, and its confidence
                        # with color depending on the object ID
                        global YOLO_DEBUG_OUTPUT_IMAGE_COUNTER
                        cv2.rectangle(image_rgb_yolo_debug, (int(x1), int(y1)), (int(x2), int(y2)), YOLO_OBJECT_ID_COLORS[object_id], 2)
                        cv2.putText(image_rgb_yolo_debug, f"ID: {object_id}", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YOLO_OBJECT_ID_COLORS[object_id], 2)
                        cv2.putText(image_rgb_yolo_debug, f"{confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    #self.printinfo("BBox: ", (x1, y1, x2, y2))
                    #self.printinfo("cam_1.depth: ", cam_1.depth.shape, np.sum((cam_1.depth > 0.001)),  cam_1.depth.dtype)

                    # Create a binary mask for the detected object of same size as RGB image
                    mask = np.zeros((image_rgb_rows, image_rgb_cols), dtype=np.uint8)

                    # Fill in a white rectangle using the (x1, y1, x2, y2) corner coordinates:
                    mask = cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)
                    
                    if DO_DEBUG_SESSION:
                        valid = (cam_1.depth>=0.001) & (mask>0)
                        self.printinfo(f"Area of valid: {valid.sum()}")

                    K     = cam_1.intrinsics.copy()
                    color = cam_1.rgb.copy()
                    depth = cam_1.depth.copy()

                    # Crop a ROI around the YOLO detection to then pass to FoundationPose estimator:
                    roi_padding = 20
                    roi_x = x1 - roi_padding
                    roi_y = y1 - roi_padding
                    roi_width = x2-x1 + 2 * roi_padding
                    roi_height = y2-y1 + 2* roi_padding
                    roi = (roi_x, roi_y, roi_width, roi_height)
                    
                    color = du.extract_roi_with_padding(color, roi)
                    depth = du.extract_roi_with_padding(depth, roi)
                    mask = du.extract_roi_with_padding(mask, roi)
                    
                    # Focal length does not change with an image crop, only the principal point.
                    # We need to translate the principal point of the camera given the ROI's coordinate system origin
                    K[0,2] -= roi_x
                    K[1,2] -= roi_y

                    # Should we reduce image working size?
                    if SHORTER_SIDE is not None and SHORTER_SIDE > 0:
                        original_shorter_side = np.min(color.shape)
                        scale_factor = SHORTER_SIDE / original_shorter_side
                        if (scale_factor < 1.0):
                            K[:2] *= scale_factor # compatible with prior principal point translation given image crop
                            color = cv2.resize(color, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                            depth = cv2.resize(depth, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                            mask  = cv2.resize(mask,  None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                        
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
                    mask = mask.astype(bool)

                    # Multiply by the depth factor (0.1) to get from raw depth pixel values to millimeters,
                    # then convert from millimeters to meters (what FoundationPose expects).
                    depth *= DEPTH_IMAGE_SCALE_PX2MM / 1000.0

                    object_pose = foundationPoseEstimator.estimate( object_class_id=object_id,
                                                                    K     = K, 
                                                                    mesh  = self.object_cad_model_cache[object_id],
                                                                    color = color,
                                                                    depth = depth,
                                                                    mask  = mask
                                                                )

                    if DO_VISUAL_DEBUG:                        
                        # Draw the 3D bounding box for this object pose on the image, given the camera pose
                        # For synthetic PBR data, we are defining camera 1 pose to be identity, hence no transformation on center_pose defined below:
                        mesh = self.object_cad_model_cache[object_id]
                        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                        center_pose = object_pose @ np.linalg.inv(to_origin)
                        original_unscaled_K = cam_1.intrinsics.copy()
                        image_rgb_yolo_debug = draw_posed_3d_box(original_unscaled_K, img=image_rgb_yolo_debug, ob_in_cam=center_pose, bbox=bbox, line_color=YOLO_OBJECT_ID_COLORS[object_id], linewidth=1)
                        image_rgb_yolo_debug = draw_xyz_axis(image_rgb_yolo_debug, ob_in_cam=center_pose, scale=0.01, K=original_unscaled_K, thickness=3, transparency=0, is_input_rgb=True)
                    
                    # Need to convert from meters (output by FoundationPose) to millimeters again
                    object_pose[0:3, 3] *= 1000.0

                    # Need to map from camera pose to world pose
                    # cam_1.pose is defined as transformation matrix world-to-camera, hence we need to invert it
                    # to become camera-to-world
                    object_pose = np.linalg.inv(cam_1.pose) @ object_pose

                    if DO_DEBUG_SESSION:
                        self.printinfo("object_pose: ")
                        print(object_pose)
                        self.printinfo("--------------------")

                    pose_estimate = PoseEstimateMsg()
                    pose_estimate.obj_id = object_id
                    pose_estimate.score = confidence
                    
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
                    if DO_DEBUG_SESSION:
                        self.printinfo(f"Foundation Pose inference time for one object instance: {inference_time:.3f} seconds")


        if DO_VISUAL_DEBUG:
            cv2.imwrite(os.path.join(YOLO_DEBUG_OUTPUT_DIR, f"yolo_debug_output_{YOLO_DEBUG_OUTPUT_IMAGE_COUNTER:06d}.png"), image_rgb_yolo_debug)
            YOLO_DEBUG_OUTPUT_IMAGE_COUNTER += 1

        return pose_estimates


def main(argv=sys.argv):
    rclpy.init(args=argv)

    pose_estimator = PoseEstimator()

    rclpy.spin(pose_estimator)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
