import cv2
import pytesseract
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os

def on_mouse(event, x, y, flags, param):
    global current_region, units_region, player_data_region, units_clicked, player_data_clicked, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if not units_clicked:
            # Place a box of size `units_template.png` at the clicked position
            units_region = (x, y, x + units_template_width, y + units_template_height)
            units_clicked = True
            print("Units region selected!")
            # Update text prompt
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Click to place the player data region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.imshow("Thumbnail", image_copy)
        elif not player_data_clicked:
            # Place a box of size `player_data_template.png` at the clicked position
            player_data_region = (x, y, x + player_data_template_width, y + player_data_template_height)
            player_data_clicked = True
            print("Player data region selected!")
            # Update text prompt
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Regions placed. Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.rectangle(image_copy, (player_data_region[0], player_data_region[1]), (player_data_region[2], player_data_region[3]), (0, 0, 255), 2)
            cv2.imshow("Thumbnail", image_copy)
            cv2.waitKey(500)  # Short delay to show the final image

def refine_bounding_box(image, region, template):
    x1, y1, x2, y2 = region
    template_height, template_width = template.shape[:2]
    
    # Extract search area
    search_area = image[y1:y2, x1:x2]
    
    # Ensure search area is larger than the template
    if search_area.shape[0] < template_height or search_area.shape[1] < template_width:
        print("Search area is smaller than template, skipping refinement.")
        return region
    
    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    refined_x1 = x1 + max_loc[0]
    refined_y1 = y1 + max_loc[1]
    refined_x2 = refined_x1 + template_width
    refined_y2 = refined_y1 + template_height
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

        # Refine selected regions using template matching
        units_region = refine_bounding_box(original_image, units_region, units_template)
        player_data_region = refine_bounding_box(original_image, player_data_region, player_data_template)

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
