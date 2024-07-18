import tkinter as tk
from tkinter import filedialog

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        print(file_path)  # For now, just print the selected file path

# Create the main window
root = tk.Tk()
root.title("Video Selector")
root.geometry("1200x800")  # Set window size

# Create a button to select the video file
select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack()

root.mainloop()
