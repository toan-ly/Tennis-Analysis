import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        
    
    def predict(self, im):
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = self.transform(im_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(im_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        orig_h, orig_w = im_rgb.shape[:2]
        
        keypoints[::2] *= orig_w / 224.0
        keypoints[1::2] *= orig_h / 224.0
        return keypoints
    
    def draw_keypoints(self, im, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.putText(im, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(im, (x, y), 5, (0, 0, 255), -1)

        return im

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

