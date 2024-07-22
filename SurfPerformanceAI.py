import cv2
import pytesseract
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variables
units_region = None
surftimer_region = None
units_clicked = False
surftimer_clicked = False
image_copy = None
original_image = None
units_template_width = 0
units_template_height = 0
surftimer_template_width = 0
surftimer_template_height = 0

def on_mouse(event, x, y, flags, param):
    global units_region, surftimer_region, units_clicked, surftimer_clicked, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if not units_clicked:
            units_width_half = units_template_width // 2
            units_height_half = units_template_height // 2
            units_x1 = x - units_width_half
            units_y1 = y - units_height_half
            units_x2 = x + units_width_half
            units_y2 = y + units_height_half

            units_region = (units_x1, units_y1, units_x2, units_y2)
            units_clicked = True
            print("Units region selected!")
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Click to place the surf timer region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.imshow("Thumbnail", image_copy)
        elif not surftimer_clicked:
            surftimer_width_half = surftimer_template_width // 2
            surftimer_height_half = surftimer_template_height // 2
            surftimer_x1 = x - surftimer_width_half
            surftimer_y1 = y - surftimer_height_half
            surftimer_x2 = x + surftimer_width_half
            surftimer_y2 = y + surftimer_height_half

            surftimer_region = (surftimer_x1, surftimer_y1, surftimer_x2, surftimer_y2)
            surftimer_clicked = True
            print("Surf timer region selected!")
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Regions placed. Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.rectangle(image_copy, (surftimer_region[0], surftimer_region[1]), (surftimer_region[2], surftimer_region[3]), (0, 0, 255), 2)
            cv2.imshow("Thumbnail", image_copy)
            cv2.waitKey(500)  # Short delay to show the final image

def refine_bounding_box(image, region):
    x1, y1, x2, y2 = region
    
    # Ensure coordinates are within image bounds
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])

    # Extract the region of interest (ROI) from the image
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        print("Error: ROI is empty")
        return region  # Return the original region if ROI is empty

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
    global video_path, thumbnail_label, units_region, surftimer_region, units_clicked, surftimer_clicked, original_image, image_copy
    global units_template_width, units_template_height, surftimer_template_width, surftimer_template_height

    units_clicked = False
    surftimer_clicked = False

    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Check if video loaded successfully
        if not cap.isOpened():
            print("Error opening video file!")
            return

        # Get the total number of frames and calculate the middle frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_number = total_frames // 2

        # Set the video position to the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

        # Read the middle frame
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
        units_template_path = os.path.join(script_dir, "data/units_template.png")
        surftimer_template_path = os.path.join(script_dir, "data/surftimer_template.png")
        
        units_template = cv2.imread(units_template_path, cv2.IMREAD_COLOR)
        surftimer_template = cv2.imread(surftimer_template_path, cv2.IMREAD_COLOR)
        
        # Verify template images are loaded
        if units_template is None:
            print(f"Error: Could not load template image from {units_template_path}")
            return
        if surftimer_template is None:
            print(f"Error: Could not load template image from {surftimer_template_path}")
            return

        # Get template dimensions
        units_template_height, units_template_width = units_template.shape[:2]
        surftimer_template_height, surftimer_template_width = surftimer_template.shape[:2]

        # Display thumbnail with mouse callback
        cv2.putText(image_copy, "Click to place the units region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Thumbnail", image_copy)
        cv2.setMouseCallback("Thumbnail", on_mouse)

        while not units_clicked:
            cv2.waitKey(1)

        while not surftimer_clicked:
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        # Refine selected regions using Tesseract OCR
        units_region = refine_bounding_box(original_image, units_region)
        surftimer_region = refine_bounding_box(original_image, surftimer_region)

        # Highlight selected regions
        image_with_regions = original_image.copy()
        cv2.rectangle(image_with_regions, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
        cv2.rectangle(image_with_regions, (surftimer_region[0], surftimer_region[1]), (surftimer_region[2], surftimer_region[3]), (0, 0, 255), 2)

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

# Create a frame to hold the button and thumbnail
frame = tk.Frame(root)
frame.pack(pady=10)

# Create a button to select the video file
select_button = tk.Button(frame, text="Select Video", command=select_video, bg="blue", fg="white", font=("Arial", 14), padx=10, pady=5)
select_button.pack()

# Create a label to display the thumbnail
thumbnail_label = tk.Label(root)
thumbnail_label.pack(fill=tk.BOTH, expand=True)

# Center the frame in the window
root.update()
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Move the button to the bottom of the window
frame.pack(side=tk.BOTTOM)

root.mainloop()
