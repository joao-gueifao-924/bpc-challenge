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
from bpc.utils.data_utils import Capture


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
OBJECT_CAD_MODEL_PATH = "/opt/ros/underlay/install/3d_models" # mounted at runtime through Docker

DO_VISUAL_DEBUG = False
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
    # print("shorter_side: ", shorter_side)
        
    
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
            print("Waiting for debugger to attach...")
            debugpy.wait_for_client()
            print("Debugger just attached. Continuing...")

        if DO_VISUAL_DEBUG:
            # Get RGB image from camera 1
            image_rgb_yolo_debug = cam_1.rgb.copy()
            # If single channel, convert to 3 channels
            if len(image_rgb_yolo_debug.shape) == 2:
                image_rgb_yolo_debug = np.tile(image_rgb_yolo_debug[:, :, None], (1, 1, 3))

        for object_id in object_ids:
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

            # Returns a dictionary mapping camera indices to a list of detections.
            # list(detections.keys())) returns [0, 1, 2]
            detections = pose_estimator._detect(capture)

            if USE_FOUNDATIONPOSE_ESTIMATOR:

                # Let's use only RGB camera 1 for now.

                image_rgb_rows, image_rgb_cols = cam_1.rgb.shape[0], cam_1.rgb.shape[1]
                
                if DO_DEBUG_SESSION:
                    print(f"total detections of object #{object_id}: {len(detections[0])}")

                for detection in detections[0]:
                    start_inference_time = time.time()
                    # 'bbox': (x1, y1, x2, y2),
                    (x1, y1, x2, y2) = detection['bbox']

                    if DO_VISUAL_DEBUG:
                        # Render the bounding box on the image, its object id, and its confidence
                        # with color depending on the object ID
                        confidence = detection['confidence']
                        global YOLO_DEBUG_OUTPUT_IMAGE_COUNTER
                        cv2.rectangle(image_rgb_yolo_debug, (int(x1), int(y1)), (int(x2), int(y2)), YOLO_OBJECT_ID_COLORS[object_id], 2)
                        cv2.putText(image_rgb_yolo_debug, f"ID: {object_id}", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YOLO_OBJECT_ID_COLORS[object_id], 2)
                        cv2.putText(image_rgb_yolo_debug, f"{confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    #print("BBox: ", (x1, y1, x2, y2))
                    #print("cam_1.depth: ", cam_1.depth.shape, np.sum((cam_1.depth > 0.001)),  cam_1.depth.dtype)

                    # Create a binary mask for the detected object of same size as RGB image
                    mask = np.zeros((image_rgb_rows, image_rgb_cols), dtype=np.uint8)

                    # Fill in a white rectangle using the (x1, y1, x2, y2) corner coordinates:
                    mask = cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)
                    
                    if DO_DEBUG_SESSION:
                        valid = (cam_1.depth>=0.001) & (mask>0)
                        print("Area of valid: ", valid.sum())

                    K     = cam_1.intrinsics.copy()
                    color = cam_1.rgb.copy()
                    depth = cam_1.depth.copy()

                    # Should we reduce image working size?
                    if SHORTER_SIDE is not None and SHORTER_SIDE > 0:
                        original_shorter_side = np.min(color.shape)
                        scale_factor = SHORTER_SIDE / original_shorter_side
                        if (scale_factor < 1.0):
                            #print("scale factor: ", scale_factor)
                            K[:2] *= scale_factor
                            color = cv2.resize(color, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                            depth = cv2.resize(depth, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                            mask  = cv2.resize(mask,  None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                        
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
                    mask = mask.astype(bool)

                    # Multiply by the depth factor (0.1) to get from raw depth pixel values to millimeters,
                    # then convert from lmilimeters to meters (what FoundationPose expects).
                    depth *= 0.1 / 1000 # TODO put this convertion elsewhere?

                    object_pose = foundationPoseEstimator.estimate( object_class_id=object_id,
                                                                    K     = K, 
                                                                    mesh  = self.object_cad_model_cache[object_id],
                                                                    color = color,
                                                                    depth = depth,
                                                                    mask  = mask
                                                                )

                    if DO_VISUAL_DEBUG:                        
                        # Draw the 3D bounding box for this object pose on the image, given the camera pose
                        #draw_bounding_box(image_rgb_yolo_debug, object_pose, cam_1.pose, cam_1.intrinsics, object_id)
                        mesh = self.object_cad_model_cache[object_id]
                        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                        center_pose = object_pose @ np.linalg.inv(to_origin)
                        original_unscaled_K = cam_1.intrinsics.copy()
                        image_rgb_yolo_debug = draw_posed_3d_box(original_unscaled_K, img=image_rgb_yolo_debug, ob_in_cam=center_pose, bbox=bbox, linewidth=1)
                        image_rgb_yolo_debug = draw_xyz_axis(image_rgb_yolo_debug, ob_in_cam=center_pose, scale=0.1, K=original_unscaled_K, thickness=3, transparency=0, is_input_rgb=True)
                    
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
