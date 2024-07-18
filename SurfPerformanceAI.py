import tkinter as tk
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk


def select_video():
    global video_path, thumbnail_label

    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if video_path:
        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Get the first frame as thumbnail
        ret, frame = cap.read()
        cap.release()

        # Convert OpenCV image to PIL format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = image.resize((300, 200))  # Adjust size as needed

        # Create a Tkinter image and label
        photo = PIL.ImageTk.PhotoImage(image)
        thumbnail_label.config(image=photo)
        thumbnail_label.image = photo  # Keep a reference

# Create the main window
root = tk.Tk()
root.title("Video Selector")
root.geometry("1200x800")

# Create a label to display the thumbnail
thumbnail_label = tk.Label(root)
thumbnail_label.pack()

# Create a button to select the video file
select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack()

root.mainloop()
