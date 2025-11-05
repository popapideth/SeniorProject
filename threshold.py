import mediapipe as mp
from utils import scaledTo

def get_thresholds(camrera_screen_width, camera_screen_height):

    ratio_w, ratio_h = scaledTo(camrera_screen_width, camera_screen_height)

    _ANGLE_HIP_VERT = {
        'STAND': (0,  10),
        'SQUATTING': (11, 80),
    }

    _ANGLE_KNEE_VERT = {
        'STAND': (0,  20),
        'SQUATTING': (21, 100),
    }
    _SQUAT_DEPTH = {
        'QUARTER': (45,  60),
        'HALF': (61, 80),
        'PARALLEL': (81,  100),
        'FULL': (101, 120),
    }
    thresholds = {
        'HIP_VERT': _ANGLE_HIP_VERT,
        'KNEE_VERT': _ANGLE_KNEE_VERT,
        'SQUAT_DEPTHS': _SQUAT_DEPTH,
        'DELTA_TRANS_KNEE_ANGLE': 2,  # deg
        'STABLE_POSE_TIME_COUNT': 0.25,  # time count
        'OFFSET_SHOULDERS_X': 40*ratio_w,
        'OFFSET_ANKLES_X': 40*ratio_w,
        'NEUTRAL_BIAS_TRUNK_TIBIA_ANGLE': 10,  # deg
        'KNEE_EXTEND_BEYOND_TOE': 15*ratio_w,

        'INACTIVE_THRESH': 15.0,
    }
    return thresholds

def get_mediapipe_pose(
                        static_image_mode = False, 
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5

                      ):
    pose = mp.solutions.pose.Pose(
                                    static_image_mode = static_image_mode,
                                    model_complexity = model_complexity,
                                    smooth_landmarks = smooth_landmarks,
                                    min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence
                                 )
    return pose