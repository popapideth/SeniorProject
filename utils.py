import os
import json
import statistics

import cv2
import numpy as np
import pandas as pd

def get_landmark_coord(pose_landmark, key, frame_width=640, frame_height=480):
    mark_x = int(pose_landmark[key].x * frame_width)
    mark_y = int(pose_landmark[key].y * frame_height)
    return np.array([mark_x, mark_y])

def get_chosen_joints_coord(mp_results, dict_features, direction, frame_width, frame_height):
    if (direction == 'nose'):
        nose_coord = get_landmark_coord(
            mp_results, dict_features[direction], frame_width, frame_height)
        return nose_coord

    elif (direction == 'left' or 'right'):
        shoulder_coord = get_landmark_coord(
            mp_results, dict_features[direction]['shoulder'], frame_width, frame_height)
        elbow_coord = get_landmark_coord(
            mp_results, dict_features[direction]['elbow'], frame_width, frame_height)
        wrist_coord = get_landmark_coord(
            mp_results, dict_features[direction]['wrist'], frame_width, frame_height)
        hip_coord = get_landmark_coord(
            mp_results, dict_features[direction]['hip'], frame_width, frame_height)
        knee_coord = get_landmark_coord(
            mp_results, dict_features[direction]['knee'], frame_width, frame_height)
        ankle_coord = get_landmark_coord(
            mp_results, dict_features[direction]['ankle'], frame_width, frame_height)
        foot_coord = get_landmark_coord(
            mp_results, dict_features[direction]['foot'], frame_width, frame_height)

        return shoulder_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord
    else:
        raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def find_angle(a, c, b=np.array([0, 0])):
    p1_ref = a - b
    p2_ref = c - b

    cos_theta = (np.dot(p1_ref, p2_ref)) / \
        (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # rad unit

    degree = int(180 / np.pi) * theta

    return int(degree)


def findDistance(coord1, coord2):
    dist = np.sqrt((coord2[0]-coord1[0])**2+(coord2[1]-coord1[1])**2)
    return int(dist)


def save_keyframe_json(keyframe, filename="keyframes.jsonl", path="./data", option=0):
    # แปลงเป็น list ของ dict
    landmarks_list = []
    for lm in keyframe['landmarks']:
        landmarks_list.append({
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),
            "visibility": round(lm.visibility, 4),
        })
    data = {
        'angles': keyframe['angles'],
        'landmarks': landmarks_list,
        'frame': keyframe['frame']
    }

    if option:
        return data
    else:
        data.pop('frame')
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")


def save_keyframe_csv(keyframe, filename="keyframes_full.csv", path="./data"):
    # ดึงค่ามุม
    row = keyframe["angles"].copy()
    # ดึง landmark
    for i, lm in enumerate(keyframe["landmarks"]):
        row[f"x_{i}"] = round(lm.x, 4)
        row[f"y_{i}"] = round(lm.y, 4)
        row[f"z_{i}"] = round(lm.z, 4)
        row[f"v_{i}"] = round(lm.visibility, 4)

    # สร้างโฟลเดอร์หากไม่มี หากมีก็เลือกลงได้
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)

    # แปลงเป็น DataFrame 1 แถว
    df = pd.DataFrame([row])
    df.to_csv(filepath, mode="a", header=not os.path.exists(
        filepath), index=False)


def findModeKneeAngle(list: list):
    try:
        my_list_multimode = list
        if len(my_list_multimode):
            modes = statistics.multimode(my_list_multimode)
            # หา index ของ mode ทั้งหมด
            mode_indexes = {mode: [i for i, val in enumerate(my_list_multimode) if val == mode]
                            for mode in modes}

            # ---------------> หากมี mode มากกว่า 2 ตัว
            if len(modes) > 1:
                selected_mode = max(modes)
                selected_mode_indexes = [i for i, val in enumerate(
                    my_list_multimode) if val == selected_mode]

                modes = selected_mode
                mode_indexes = selected_mode_indexes
            # --------------->
            else:
                modes = modes[0]
                mode_indexes = [i for i, val in enumerate(
                    my_list_multimode) if val == modes]

            index_centroid = int(len(mode_indexes)/2)
            mode_index_centroid = mode_indexes[index_centroid]

            return mode_index_centroid
        else:
            print("ไม่มีสมาชิกใน List")
            return None
    except Exception as e:
        print(f"แตกที่ mode {e}")


def _show_feedback(frame, point_of_mistake, display_depth, mistake_dict_maps, squat_depth_dict_map, hasINCORRECT_POSTURE):

    h, w = frame.shape[:2]
    ratio_w, ratio_h = scaledTo(w, h)

    try:
        if hasINCORRECT_POSTURE:
            for idx in np.where(point_of_mistake)[0]:
                draw_text(
                    frame,
                    mistake_dict_maps[idx][0],
                    pos=(int(30*ratio_w),
                        int(mistake_dict_maps[idx][1]*ratio_h)),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=mistake_dict_maps[idx][2]
                )

            for idx in np.where(display_depth)[0]:
                draw_text(
                    frame,
                    squat_depth_dict_map[idx][0],
                    pos=(int(30*ratio_w),
                        int(squat_depth_dict_map[idx][1]*ratio_h)),
                    text_color=(255, 255, 230),
                    font_scale=0.6,
                    text_color_bg=squat_depth_dict_map[idx][2]
                )
    except:
        print("แตกใน show_feedback")

    return frame


def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)

    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle=0, startAngle=-90, endAngle=-180, color=box_color, thickness=-1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle=0, startAngle=0, endAngle=-90, color=box_color, thickness=-1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle=0, startAngle=90, endAngle=180, color=box_color, thickness=-1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle=0, startAngle=0, endAngle=90, color=box_color, thickness=-1)
    return img


def draw_text(
    img,
    msg,
    width=7,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    pos=(0, 0),
    font_scale=1,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
    box_offset=(20, 10),
    overlay_image=False,
    overlay_type=None
):
    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size

    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n,
                    o in zip((x + text_w, y + text_h), offset, (25, 0)))

    resize_height = 0
    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)

    cv2.putText(
        img,
        msg,
        (int(rec_start[0]+resize_height + 8),
        int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    return text_size

def cropEasy(frame, point1, point2=False):
    h, w = frame.shape[:2]
    ratio_w, ratio_h = scaledTo(w, h)
    CONST_PIXEL = 50
    # cv2.imwrite("hahaha.png", frame)
    x1, y1 = point1  # Mediapipe Normalize
    if point2:
        x2, y2 = point2
        x_min = (min(x1, x2) * w - CONST_PIXEL*ratio_w) if (min(x1, x2)
                                                            * w - CONST_PIXEL*ratio_w) > 0 else 0
        x_max = (max(x1, x2) * w + CONST_PIXEL*ratio_w) if (max(x1, x2)
                                                            * w + CONST_PIXEL*ratio_w) <= w else w
        y_min = (min(y1, y2) * h - CONST_PIXEL*ratio_h) if (min(y1, y2)
                                                            * h - CONST_PIXEL*ratio_h) > 0 else 0
        y_max = (max(y1, y2) * h + CONST_PIXEL*ratio_h) if (max(y1, y2)
                                                            * h + CONST_PIXEL*ratio_h) <= h else h
        crop_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        # cv2.imwrite("hahaha2.png", crop_img)
        return crop_img

    x_min = (x1 * w * ratio_w) - (CONST_PIXEL*ratio_w) if (x1 *
                                                        w * ratio_w) + (CONST_PIXEL*ratio_w) > 0 else 0
    x_max = (x1 * w * ratio_w) + (CONST_PIXEL*ratio_w) if (x1 *
                                                        w * ratio_w) + (CONST_PIXEL*ratio_w) <= w else w
    y_min = (y1 * h * ratio_h) - (CONST_PIXEL*ratio_h) if (y1 *
                                                        h * ratio_h) - (CONST_PIXEL*ratio_h) > 0 else 0
    y_max = (y1 * h * ratio_h) + (CONST_PIXEL*ratio_h) if (y1 *
                                                        h * ratio_h) + (CONST_PIXEL*ratio_h) <= h else h
    crop_img = frame[int(y_min): int(y_max),
                    int(x_min): int(x_max)]
    return crop_img


def overlayImage(background, overlay):

    # Get dimension
    h, w, _ = background.shape
    oh, ow, _ = overlay.shape
    ratio_w, ratio_h = scaledTo(w, h)

    x_offset = int(h*0.05)
    y_offset = int(w*0.05)

    # Ensure overlay fits within background
    if x_offset + ow <= w and y_offset + oh <= h:
        # Extract alpha channel if present in overlay
        if overlay.shape[2] == 4:
            alpha_channel = overlay[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_channel

            for c in range(0, 3):
                background[y_offset:y_offset+oh, x_offset:x_offset+ow, c] = \
                    (alpha_inv * background[y_offset:y_offset+oh, x_offset:x_offset+ow, c] +
                    alpha_channel * overlay[:, :, c])
        else:
            # Simple overlay for non-transparent images
            background[y_offset:y_offset+oh, x_offset:x_offset+ow] = overlay

        return background
        # cv2.imshow("Overlayed Image", background)
    else:
        cv2.putText(background, "No Overlay",
                    (50, 50), 1, 1, (0, 0, 255), 1, 1)
        return background

def baseWidthHeight():
    base_w, base_h = 640, 480
    return base_w, base_h

def scaledTo(width, height):
    old_w, old_h = baseWidthHeight()

    # จอใหม่
    new_w, new_h = width, height

    # คิดอัตราส่วน
    display_scale_x = new_w / old_w
    display_scale_y = new_h / old_h

    return display_scale_x, display_scale_y