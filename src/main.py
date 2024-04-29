from utils import (read_video, 
                   save_video,
                   measure_dist,
                   draw_player_stats,
                   convert_px_to_meter)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd

def main():
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl'
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/ball_detections.pkl'
                                                     )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector Model
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # Choose players
    player_detections = player_tracker.filter_players(court_keypoints, player_detections)
    
    # Mini court
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bbox_to_mini_court_coord(player_detections,
                                                                                                          ball_detections,
                                                                                                          court_keypoints)
    player_stats_data = [{
        'frame_num':0,
        'player_1_num_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_num_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    }]

    for ball_shot_idx in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_idx]
        end_frame = ball_shot_frames[ball_shot_idx + 1]
        ball_shot_time_sec = (end_frame - start_frame) / 24 # 24fps

        # Get distance covered by ball
        dist_covered_by_ball_pixels = measure_dist(ball_mini_court_detections[start_frame][1],
                                                   ball_mini_court_detections[end_frame][1])
        dist_covered_by_ball_meters = convert_px_to_meter(dist_covered_by_ball_pixels,
                                                          constants.DOUBLE_LINE_WIDTH,
                                                          mini_court.get_width_of_mini_court())

        # Speed of ball shot in km/h
        speed_ball_shot = dist_covered_by_ball_meters / ball_shot_time_sec * 3.6

        # Player hitting ball
        player_positions = player_mini_court_detections[start_frame]
        player_hit = min(player_positions.keys(), key=lambda player_id: measure_dist(player_positions[player_id],
                                                                                     ball_mini_court_detections[start_frame][1]))

        # Opponent speed
        opponent_id = 1 if player_hit == 2 else 2
        dist_covered_by_opponent_pixels = measure_dist(player_mini_court_detections[start_frame][opponent_id],
                                                       player_mini_court_detections[end_frame][opponent_id])
        dist_covered_by_opponent_meters = convert_px_to_meter(dist_covered_by_opponent_pixels,
                                                              constants.DOUBLE_LINE_WIDTH,
                                                              mini_court.get_width_of_mini_court())

        opponent_speed = dist_covered_by_opponent_meters / ball_shot_time_sec * 3.6

        curr_player_stats = deepcopy(player_stats_data[-1])
        curr_player_stats['frame_num'] = start_frame
        curr_player_stats[f'player_{player_hit}_num_shots'] += 1
        curr_player_stats[f'player_{player_hit}_total_shot_speed'] += speed_ball_shot
        curr_player_stats[f'player_{player_hit}_last_shot_speed'] = speed_ball_shot

        curr_player_stats[f'player_{opponent_id}_total_player_speed'] += opponent_speed
        curr_player_stats[f'player_{opponent_id}_last_player_speed'] = opponent_speed

        player_stats_data.append(curr_player_stats)

    stats_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    stats_df = pd.merge(frames_df, stats_df, on='frame_num', how='left')
    stats_df = stats_df.ffill()

    stats_df['player_1_average_shot_speed'] = stats_df['player_1_total_shot_speed']/stats_df['player_1_num_shots']
    stats_df['player_2_average_shot_speed'] = stats_df['player_2_total_shot_speed']/stats_df['player_2_num_shots']
    stats_df['player_1_average_player_speed'] = stats_df['player_1_total_player_speed']/stats_df['player_2_num_shots']
    stats_df['player_2_average_player_speed'] = stats_df['player_2_total_player_speed']/stats_df['player_1_num_shots']

    # Draw output
    
    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    # Draw player stats
    output_video_frames = draw_player_stats(output_video_frames, stats_df)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()