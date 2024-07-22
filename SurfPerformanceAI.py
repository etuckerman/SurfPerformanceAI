import cv2
import pytesseract
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def on_mouse(event, x, y, flags, param):
    global current_region, units_region, player_data_region, units_clicked, player_data_clicked, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if not units_clicked:
            # Calculate the top-left and bottom-right coordinates of the box
            units_width_half = units_template_width // 2
            units_height_half = units_template_height // 2
            units_x1 = x - units_width_half
            units_y1 = y - units_height_half
            units_x2 = x + units_width_half
            units_y2 = y + units_height_half

            # Place a box of size `units_template.png` at the clicked position
            units_region = (units_x1, units_y1, units_x2, units_y2)
            units_clicked = True
            print("Units region selected!")
            # Update text prompt
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Click to place the player data region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.imshow("Thumbnail", image_copy)
        elif not player_data_clicked:
            # Calculate the top-left and bottom-right coordinates of the box
            player_data_width_half = player_data_template_width // 2
            player_data_height_half = player_data_template_height // 2
            player_data_x1 = x - player_data_width_half
            player_data_y1 = y - player_data_height_half
            player_data_x2 = x + player_data_width_half
            player_data_y2 = y + player_data_height_half

            # Place a box of size `player_data_template.png` at the clicked position
            player_data_region = (player_data_x1, player_data_y1, player_data_x2, player_data_y2)
            player_data_clicked = True
            print("Player data region selected!")
            # Update text prompt
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Regions placed. Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.rectangle(image_copy, (player_data_region[0], player_data_region[1]), (player_data_region[2], player_data_region[3]), (0, 0, 255), 2)
            cv2.imshow("Thumbnail", image_copy)
            cv2.waitKey(500)  # Short delay to show the final image

def refine_bounding_box(image, region):
    x1, y1, x2, y2 = region
    
    # Extract the region of interest (ROI) from the image
    roi = image[y1:y2, x1:x2]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to detect text boxes within the ROI
    boxes = pytesseract.image_to_boxes(gray)
    
    # Initialize variables to store the refined bounding box coordinates
    refined_x1, refined_y1, refined_x2, refined_y2 = x2, y2, x1, y1
    
    for b in boxes.splitlines():
        b = b.split()
        b_x1, b_y1, b_x2, b_y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        
        # Tesseract coordinates need to be adjusted to the image coordinates
        b_x1 += x1
        b_y1 = y2 - b_y1
        b_x2 += x1
        b_y2 = y2 - b_y2
        
        # Update the refined bounding box coordinates
        refined_x1 = min(refined_x1, b_x1)
        refined_y1 = min(refined_y1, b_y2)
        refined_x2 = max(refined_x2, b_x2)
        refined_y2 = max(refined_y2, b_y1)
    
    refined_region = (refined_x1, refined_y1, refined_x2, refined_y2)
    return refined_region

def select_video():
    global video_path, thumbnail_label, units_region, player_data_region, units_clicked, player_data_clicked, original_image, image_copy
    global units_template_width, units_template_height, player_data_template_width, player_data_template_height

    units_clicked = False
    player_data_clicked = False

    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Check if video loaded successfully
        if not cap.isOpened():
            print("Error opening video file!")
            return

        # Get the first frame as thumbnail
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error reading frame from video!")
            return

        # Convert OpenCV image to PIL format
        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(original_image)
        original_image = original_image.resize((1200, 800))  # Resize to 1200x800

        # Convert PIL image back to OpenCV format (NumPy array)
        original_image = np.array(original_image)
        image_copy = original_image.copy()

        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load template images using absolute paths
        units_template_path = os.path.join(script_dir, "units_template.png")
        player_data_template_path = os.path.join(script_dir, "player_data_template.png")
        
        units_template = cv2.imread(units_template_path, cv2.IMREAD_COLOR)
        player_data_template = cv2.imread(player_data_template_path, cv2.IMREAD_COLOR)
        
        # Verify template images are loaded
        if units_template is None:
            print(f"Error: Could not load template image from {units_template_path}")
            return
        if player_data_template is None:
            print(f"Error: Could not load template image from {player_data_template_path}")
            return

        # Get template dimensions
        units_template_height, units_template_width = units_template.shape[:2]
        player_data_template_height, player_data_template_width = player_data_template.shape[:2]

        # Display thumbnail with mouse callback
        cv2.putText(image_copy, "Click to place the units region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Thumbnail", image_copy)
        cv2.setMouseCallback("Thumbnail", on_mouse)

        while not units_clicked:
            cv2.waitKey(1)

        while not player_data_clicked:
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        # Refine selected regions using Tesseract OCR
        units_region = refine_bounding_box(original_image, units_region)
        player_data_region = refine_bounding_box(original_image, player_data_region)

        # Highlight selected regions
        image_with_regions = original_image.copy()
        cv2.rectangle(image_with_regions, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
        cv2.rectangle(image_with_regions, (player_data_region[0], player_data_region[1]), (player_data_region[2], player_data_region[3]), (0, 0, 255), 2)

        # Convert to PIL format for Tkinter
        image_with_regions = Image.fromarray(image_with_regions)
        image_tk = ImageTk.PhotoImage(image_with_regions)

        # Update the thumbnail label in the main window
        thumbnail_label.config(image=image_tk)
        thumbnail_label.image = image_tk

# Create the main window
root = tk.Tk()
root.title("Video Selector")
root.geometry("1600x1200")

# Create a label to display the thumbnail
thumbnail_label = tk.Label(root)
thumbnail_label.pack(fill=tk.BOTH, expand=True)

# Create a button to select the video file
select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack()

root.mainloop()
