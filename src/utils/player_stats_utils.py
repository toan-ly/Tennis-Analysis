import numpy as np
import cv2

def draw_player_stats(out_frames, stats):
    for i, row in stats.iterrows():
        p1_shot_speed = row['player_1_last_shot_speed']
        p2_shot_speed = row['player_2_last_shot_speed']
        p1_speed = row['player_1_last_player_speed']
        p2_speed = row['player_2_last_player_speed']

        avg_p1_shot_speed = row['player_1_average_shot_speed']
        avg_p2_shot_speed = row['player_2_average_shot_speed']
        avg_p1_speed = row['player_1_average_player_speed']
        avg_p2_speed = row['player_2_average_player_speed']

        frame = out_frames[i]
        shapes = np.zeros_like(frame, np.uint8)

        w, h = 350, 230

        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + w
        end_y = start_y + h

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out_frames[i] = frame
        
        text = "     Player 1     Player 2"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 80, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 10, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{p1_shot_speed:.1f} km/h    {p2_shot_speed:.1f} km/h"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 130, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 10, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{p1_speed:.1f} km/h    {p2_speed:.1f} km/h"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 130, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 10, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_p1_shot_speed:.1f} km/h    {avg_p2_shot_speed:.1f} km/h"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 130, start_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 10, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_p1_speed:.1f} km/h    {avg_p2_speed:.1f} km/h"
        out_frames[i] = cv2.putText(out_frames[i], text, (start_x + 130, start_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return out_frames

