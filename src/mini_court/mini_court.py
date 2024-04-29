import cv2
import sys
sys.path.append('../')
import constants
from utils import(
    convert_meter_to_px,
    convert_px_to_meter,
    get_foot_position,
    get_closest_keypoint_idx,
    get_height_bbox,
    measure_xy_dist, 
    get_center_position, measure_dist
)
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()
        
    def set_court_lines(self):
        self.lines = [
            (0, 2), 
            (4, 5),
            (6, 7), 
            (1, 3),
            
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3)
        ]
        
    def convert_m_to_px_helper(self, meters):
        return convert_meter_to_px(meters,
                                   constants.DOUBLE_LINE_WIDTH,
                                   self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_kps = [0]*28
        
        # p0
        drawing_kps[0], drawing_kps[1] = int(self.court_start_x), int(self.court_start_y)
        # p1
        drawing_kps[2], drawing_kps[3] = int(self.court_end_x), int(self.court_start_y)
        # p2
        drawing_kps[4] = int(self.court_start_x)
        drawing_kps[5] = self.court_start_y + self.convert_m_to_px_helper(constants.HALF_COURT_LINE_HEIGHT * 2)
        # p3
        drawing_kps[6] = drawing_kps[0] + self.court_drawing_width
        drawing_kps[7] = drawing_kps[5]
        # p4
        drawing_kps[8] = drawing_kps[0] + self.convert_m_to_px_helper(constants.DOUBLE_ALLY_DIFF)
        drawing_kps[9] = drawing_kps[1]
        # p5
        drawing_kps[10] = drawing_kps[4] + self.convert_m_to_px_helper(constants.DOUBLE_ALLY_DIFF)
        drawing_kps[11] = drawing_kps[5] 
        # #p6
        drawing_kps[12] = drawing_kps[2] - self.convert_m_to_px_helper(constants.DOUBLE_ALLY_DIFF)
        drawing_kps[13] = drawing_kps[3] 
        # #p7
        drawing_kps[14] = drawing_kps[6] - self.convert_m_to_px_helper(constants.DOUBLE_ALLY_DIFF)
        drawing_kps[15] = drawing_kps[7] 
        # #p8
        drawing_kps[16] = drawing_kps[8] 
        drawing_kps[17] = drawing_kps[9] + self.convert_m_to_px_helper(constants.NO_MANS_LAND_HEIGHT)
        # p9
        drawing_kps[18] = drawing_kps[16] + self.convert_m_to_px_helper(constants.SINGLE_LINE_WIDTH)
        drawing_kps[19] = drawing_kps[17] 
        # p10
        drawing_kps[20] = drawing_kps[10] 
        drawing_kps[21] = drawing_kps[11] - self.convert_m_to_px_helper(constants.NO_MANS_LAND_HEIGHT)
        # p11
        drawing_kps[22] = drawing_kps[20] +  self.convert_m_to_px_helper(constants.SINGLE_LINE_WIDTH)
        drawing_kps[23] = drawing_kps[21] 
        # p12
        drawing_kps[24] = int((drawing_kps[16] + drawing_kps[18])/2)
        drawing_kps[25] = drawing_kps[17] 
        # p13
        drawing_kps[26] = int((drawing_kps[20] + drawing_kps[22])/2)
        drawing_kps[27] = drawing_kps[21] 

        self.drawing_key_points = drawing_kps

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
    
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out_frame = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out_frame

    def draw_mini_court(self, frames):
        out_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            out_frames.append(frame)
        return out_frames

    def draw_court(self, frame):
        # Draw points
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2 + 1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coord(self, 
                             object_pos, 
                             closest_keypoint, 
                             closest_keypoint_idx, 
                             player_height_in_pixels,
                             player_height_in_meters
                             ):
        dist_keypoint_x_pixels, dist_keypoint_y_pixels = measure_xy_dist(object_pos, closest_keypoint)

        # Convert pixel distance to meters
        dist_keypoint_x_meters = convert_px_to_meter(dist_keypoint_x_pixels,
                                                     player_height_in_meters,
                                                     player_height_in_pixels)
        dist_keypoint_y_meters = convert_px_to_meter(dist_keypoint_y_pixels, 
                                                     player_height_in_meters,
                                                     player_height_in_pixels)

        # Convert to mini court coord
        mini_court_x_dist_pixels = self.convert_m_to_px_helper(dist_keypoint_x_meters)
        mini_court_y_dist_pixels = self.convert_m_to_px_helper(dist_keypoint_y_meters)
        closest_mini_court_keypoint = (self.drawing_key_points[closest_keypoint_idx * 2],
                                       self.drawing_key_points[closest_keypoint_idx * 2 + 1])

        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_dist_pixels,
                                      closest_mini_court_keypoint[1] + mini_court_y_dist_pixels)

        return mini_court_player_position

    def convert_bbox_to_mini_court_coord(self, player_boxes, ball_boxes, orig_court_kps):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }    

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_position(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_dist(ball_position, get_center_position(player_bbox[x])))
            
            output_player_boxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest keypoints in pixels
                closest_keypoint_idx = get_closest_keypoint_idx(foot_position, orig_court_kps, [0, 2, 12, 13])
                closest_keypoint = (orig_court_kps[closest_keypoint_idx * 2],
                                    orig_court_kps[closest_keypoint_idx * 2 + 1])

                # Get player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bboxes_heights_in_pixels = [get_height_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_pos = self.get_mini_court_coord(foot_position,
                                                                  closest_keypoint,
                                                                  closest_keypoint_idx,
                                                                  max_player_height_in_pixels, 
                                                                  player_heights[player_id]
                                                                  )


                output_player_boxes_dict[player_id] = mini_court_player_pos
                
                if closest_player_id_to_ball == player_id:
                    # Get the closest keypoints in pixels
                    closest_keypoint_idx = get_closest_keypoint_idx(ball_position, orig_court_kps, [0, 2, 12, 13])
                    closest_keypoint = (orig_court_kps[closest_keypoint_idx * 2],
                                    orig_court_kps[closest_keypoint_idx * 2 + 1]) 

                    mini_court_player_pos = self.get_mini_court_coord(ball_position,
                                                                      closest_keypoint,
                                                                      closest_keypoint_idx,
                                                                      max_player_height_in_pixels,
                                                                      player_heights[player_id])

                    output_ball_boxes.append({1: mini_court_player_pos})
            output_player_boxes.append(output_player_boxes_dict)
        return output_player_boxes, output_ball_boxes
                
    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, pos in positions[frame_num].items():
                x, y = int(pos[0]), int(pos[1])
                cv2.circle(frame, (x, y), 5, color, -1)
        return frames
