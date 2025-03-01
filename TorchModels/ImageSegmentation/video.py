import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import NCA

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = NCA(input_channels=3, state_channels=64, hidden_channels=64, output_channels=3).to(device)
model.load_state_dict(torch.load('higher_res_robust_using350350_20epochs.pth', weights_only=True))
model.eval()
                                         
# Define the transformation for input frames
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))  # Resize frames to 64x64
])

# Define the inverse transformation to resize back to original 
inverse_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64))  # Resize frames to 350x350
])

def process_frame(frame):
    # Convert the frame to PIL Image and apply transformations
    frame_img = Image.fromarray(frame)
    img = transform(frame_img).unsqueeze(0).to(device)
    # Process with the model
    with torch.no_grad():
        output = model.forward(img)
        output_classes = torch.argmax(output, dim=1).squeeze(0)
    
    # Convert the output to a binary mask
    output_classes_np = output_classes.cpu().numpy()
    mask = np.zeros_like(output_classes_np, dtype=np.uint8)
    mask[output_classes_np == 1] = 255  # Example: Class 1 target

    # Resize mask back to original size
    mask_img = Image.fromarray(mask).resize(frame.shape[:2][::-1], resample=Image.NEAREST)
    mask = np.array(mask_img)

    # Create a green overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[mask > 0] = [0, 255, 0]  # Green color
    
    # Blend the overlay with the original frame
    alpha = 0.5  # Adjust the transparency level (0 = fully transparent, 1 = fully opaque)
    blended_frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    print("Processed Frame")
    return blended_frame

def main(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video. Check file format and path.")
        print(f"VideoCapture status: {cap.isOpened()}")
        exit()
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Open a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Write the processed frame to the output video
        out.write(processed_frame)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    input_video_path = 'input_video.mp4'
    output_video_path = 'output_video.mp4'
    main(input_video_path, output_video_path)
