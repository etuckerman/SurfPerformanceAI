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

        # Get the first frame as thumbnail
        ret, frame = cap.read()
        cap.release()

        # Convert OpenCV image to PIL format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((1200, 800))  # Resize to 1200x800

        # Create a Tkinter image and label
        photo = ImageTk.PhotoImage(image)
        thumbnail_label.config(image=photo)
        thumbnail_label.image = photo  # Keep a reference

        # Display thumbnail with mouse callback
        cv2.imshow("Thumbnail", np.array(image))
        cv2.setMouseCallback("Thumbnail", on_mouse)
        cv2.waitKey(0)  # Wait for user input before closing
        cv2.destroyAllWindows()

        # Create a Tkinter image and label
        photo = ImageTk.PhotoImage(image)
        thumbnail_label.config(image=photo)
        thumbnail_label.image = photo  # Keep a reference

        # Process video
        cap = cv2.VideoCapture(video_path)

        with open("data.csv", "w", newline="") as csvfile:
            fieldnames = ["Units", "Player Data"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract data only if both regions are defined
                if units_clicked and player_data_clicked:
                    # Extract units data
                    units_roi = frame[units_region[1]:units_region[3], units_region[0]:units_region[2]]
                    gray_units = cv2.cvtColor(units_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh_units = cv2.threshold(gray_units, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    units_text = pytesseract.image_to_string(thresh_units, config='--psm 6')
                    units_text = units_text.strip()  # Remove extra spaces

                    # Extract player data
                    player_data_roi = frame[player_data_region[1]:player_data_region[3], player_data_region[0]:player_data_region[2]]
                    gray_player_data = cv2.cvtColor(player_data_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh_player_data = cv2.threshold(gray_player_data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    player_data_text = pytesseract.image_to_string(thresh_player_data)
                    player_data_text = player_data_text.strip()

                    # Write data to CSV
                    writer.writerow({"Units": units_text, "Player Data": player_data_text})

        cap.release()
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
