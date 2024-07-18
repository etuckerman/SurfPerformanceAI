import cv2
import pytesseract
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import csv


def on_mouse(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, current_region, units_clicked, player_data_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        if not units_clicked:
            current_region = "units"
        elif not player_data_clicked:
            current_region = "player_data"

    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        if current_region == "units":
            units_region = (start_x, start_y, end_x, end_y)
            units_clicked = True
            print("Units region selected!")
        elif current_region == "player_data":
            player_data_region = (start_x, start_y, end_x, end_y)
            player_data_clicked = True
            print("Player data region selected!")
            
def detect_text(image, region):
    # Extract ROI from the image
    roi = image[region[1]:region[3], region[0]:region[2]]

    # Preprocess image for text detection (optional)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Use PyTesseract to detect text
    text = pytesseract.image_to_string(thresh)
    return text

def select_video():
    global video_path, thumbnail_label, units_region, player_data_region, units_clicked, player_data_clicked

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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((1200, 800))  # Resize to 1200x800

        # Convert PIL image back to OpenCV format (NumPy array)
        image = np.array(image)
        
        # Display thumbnail with mouse callback
        cv2.imshow("Thumbnail", np.array(image))
        cv2.putText(image, "Click the units region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add prompt
        cv2.waitKey(1)  # Short delay for image update
        cv2.setMouseCallback("Thumbnail", on_mouse)

        while not units_clicked:
            cv2.waitKey(1)

        cv2.putText(image, "Click the player data region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Update prompt
        cv2.imshow("Thumbnail", np.array(image))
        cv2.waitKey(1)

        while not player_data_clicked:
            cv2.waitKey(1)

        cv2.destroyAllWindows()


# Create the main window
root = tk.Tk()
root.title("Video Selector")
root.geometry("1200x800")

# Create a label to display the thumbnail
thumbnail_label = tk.Label(root)
thumbnail_label.pack(fill=tk.BOTH, expand=True)

# Create a button to select the video file
select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack()

root.mainloop()
