import cv2
import pytesseract
from tkinter import filedialog, messagebox
import tkinter as tk
import tkvideo
from PIL import Image, ImageTk
import numpy as np
import os
import re
import time
import csv


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def write_to_csv(data, csv_file_path):
    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Frame Number', 'Units Data', 'Surf Timer Data'])
        # Write the data
        for row in data:
            writer.writerow(row)

def on_mouse(event, x, y, flags, param):
    global units_region, surftimer_region, units_clicked, surftimer_clicked, image_copy

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
            # Add a small gap to the region
            gap = 5
            units_region = (units_x1 - gap, units_y1 - gap, units_x2 + gap, units_y2 + gap)
            units_clicked = True
            print("Units region selected!")
            # Update text prompt
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Click to place the surf timer region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.imshow("Thumbnail", image_copy)
        elif not surftimer_clicked:
            # Calculate the top-left and bottom-right coordinates of the box
            surftimer_width_half = surftimer_template_width // 2
            surftimer_height_half = surftimer_template_height // 2
            surftimer_x1 = x - surftimer_width_half
            surftimer_y1 = y - surftimer_height_half
            surftimer_x2 = x + surftimer_width_half
            surftimer_y2 = y + surftimer_height_half

            # Place a box of size `surftimer_template.png` at the clicked position
            gap = 5
            surftimer_region = (surftimer_x1 - gap, surftimer_y1 - gap, surftimer_x2 + gap, surftimer_y2 + gap)
            surftimer_clicked = True
            print("Surf timer region selected!")
            # Update text prompt
            image_copy = original_image.copy()
            cv2.putText(image_copy, "Regions placed. Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
            cv2.rectangle(image_copy, (surftimer_region[0], surftimer_region[1]), (surftimer_region[2], surftimer_region[3]), (0, 0, 255), 2)
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

def extract_text_from_box(image, region, scale_factor=2):
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
        return ""
    
    # Invert the colors of the ROI
    inverted_roi = cv2.bitwise_not(roi)
    
    # Resize the inverted ROI to improve text detection
    height, width = inverted_roi.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_roi = cv2.resize(inverted_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Use Tesseract to extract text from the resized ROI
    text = pytesseract.image_to_string(resized_roi, config = '--psm 6')
    
    # Print the image for debugging
    #cv2.imshow("Original ROI", roi)
    #cv2.imshow("Inverted ROI", inverted_roi)
    #cv2.imshow("Resized ROI", resized_roi)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    print(f"Extracted text: {text}")  # Debug print
    return text




def crop_video(video_path, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_path)
    
    # Check if video loaded successfully
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Convert start and end times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    # Set the video position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Video cropped and saved to {output_path}")


def extract_data_from_spread_out_frames(video_path, units_region, surftimer_region, csv_file_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames
    interval = max(total_frames // num_frames, 1)
    
    frame_number = 0
    data = []

    for i in range(num_frames):
        # Set the video position to the next frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract text from the units region
        units_text = extract_text_from_box(frame, units_region)
        
        # Extract text from the surf timer region
        surf_timer_text = extract_text_from_box(frame, surftimer_region)
        
        # Append the data to the list
        data.append([frame_number, units_text, surf_timer_text])
        
        frame_number += interval
    
    cap.release()
    
    # Write data to CSV
    write_to_csv(data, csv_file_path)
    print(f"Data saved to {csv_file_path}")

def scale_regions(original_frame_size, resized_frame_size, regions):
    original_width, original_height = original_frame_size
    resized_width, resized_height = resized_frame_size
    
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    
    scaled_regions = []
    for region in regions:
        x1, y1, x2, y2 = region
        scaled_region = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )
        scaled_regions.append(scaled_region)
    
    return scaled_regions

def process_video_frames(video_path, units_region, surftimer_region, csv_file_path, text_label):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    # Get original frame size
    original_frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get resized frame size
        resized_frame_size = (1200, 800)
        frame_resized = cv2.resize(frame, resized_frame_size)
        
        # Scale the regions to match the resized frame
        scaled_units_region, scaled_surftimer_region = scale_regions(original_frame_size, resized_frame_size, [units_region, surftimer_region])
        
        # Extract text from the resized regions
        units_text = extract_text_from_box(frame_resized, scaled_units_region)
        surf_timer_text = extract_text_from_box(frame_resized, scaled_surftimer_region)
        
        # Append the data to the list
        data.append([frame_number, units_text, surf_timer_text])
        
        # Draw rectangles on the resized frame
        frame_with_regions = frame_resized.copy()
        cv2.rectangle(frame_with_regions, (scaled_units_region[0], scaled_units_region[1]), (scaled_units_region[2], scaled_units_region[3]), (0, 255, 0), 2)
        cv2.rectangle(frame_with_regions, (scaled_surftimer_region[0], scaled_surftimer_region[1]), (scaled_surftimer_region[2], scaled_surftimer_region[3]), (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Video Frame", frame_with_regions)
        
        # Update the Tkinter label with extracted text
        text_label.config(text=f"Frame {frame_number}: Units Text: {units_text}\nSurf Timer Text: {surf_timer_text}")
        root.update_idletasks()  # Update Tkinter window
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Write data to CSV
    write_to_csv(data, csv_file_path)
    print(f"Data saved to {csv_file_path}")






def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Check if video loaded successfully
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow("Video", frame)
        
        # Check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_video_frames_with_thumbnail_regions(video_path, units_region, surftimer_region, csv_file_path, text_label):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to match the thumbnail size
        resized_frame = cv2.resize(frame, (1200, 800))
        
        # Use the same regions as in the thumbnail
        # Extract text from the regions directly
        units_text = extract_text_from_box(resized_frame, units_region)
        surf_timer_text = extract_text_from_box(resized_frame, surftimer_region)
        
        # Append the data to the list
        data.append([frame_number, units_text, surf_timer_text])
        
        # Draw rectangles on the resized frame
        frame_with_regions = resized_frame.copy()
        cv2.rectangle(frame_with_regions, (units_region[0], units_region[1]), (units_region[2], units_region[3]), (0, 255, 0), 2)
        cv2.rectangle(frame_with_regions, (surftimer_region[0], surftimer_region[1]), (surftimer_region[2], surftimer_region[3]), (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Video Frame", frame_with_regions)
        
        # Update the Tkinter label with extracted text
        text_label.config(text=f"Frame {frame_number}: Units Text: {units_text}\nSurf Timer Text: {surf_timer_text}")
        root.update_idletasks()  # Update Tkinter window
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Write data to CSV
    write_to_csv(data, csv_file_path)
    print(f"Data saved to {csv_file_path}")

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

        # Add some room around the refined regions
        units_region = (units_region[0] - 10, units_region[1] - 10, units_region[2] + 10, units_region[3] + 10)
        surftimer_region = (surftimer_region[0] - 10, surftimer_region[1] - 10, surftimer_region[2] + 10, surftimer_region[3] + 10)

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

        # Ask user to confirm selected regions
        confirm = messagebox.askyesno("Confirm Regions", "Are the selected regions correct?")
        if not confirm:
            print("User canceled the region selection.")
            return

        # Prompt user to select CSV file path
        csv_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if csv_file_path:
            # Process the video frames and save data to CSV
            process_video_frames_with_thumbnail_regions(video_path, units_region, surftimer_region, csv_file_path, text_label)
        
        # Play the video in the main Tkinter window
        play_video(video_path)

# Create the main window
root = tk.Tk()
root.title("Video Selector")
root.geometry("1200x800")

# Create a frame to hold the button and thumbnail
frame = tk.Frame(root)
frame.pack(pady=10)

# Create a button to select the video file
select_button = tk.Button(frame, text="Select Video", command=select_video, bg="blue", fg="white", font=("Arial", 14), padx=10, pady=5)
select_button.pack()

# Create a label to display the thumbnail
thumbnail_label = tk.Label(root)
thumbnail_label.pack(fill=tk.BOTH, expand=True)

# Create a label to display extracted text
text_label = tk.Label(root, text="", font=("Arial", 12))
text_label.pack(pady=10)

# Center the frame in the window
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Move the button to the bottom of the window
frame.pack(side=tk.BOTTOM)

# Start the Tkinter event loop
root.mainloop()

