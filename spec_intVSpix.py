import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # change the parameter to access your particular webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame.")
        return None
    
    return frame

# Function to process the captured image to extract the spectrum
def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sum the pixel values along the vertical axis to get the intensity distribution
    intensity = np.sum(gray, axis=0)
    
    return intensity

# Function to plot the spectrum as intensity vs. pixel position
def plot_spectrum(intensity):
    pixel_positions = np.arange(len(intensity))
    
    plt.plot(pixel_positions, intensity)
    plt.xlabel('Pixel Position')
    plt.ylabel('Intensity')
    plt.title('Spectrum: Intensity vs. Pixel Position')
    plt.show()

# Main function to capture, process and plot the spectrum
def main():
    image = capture_image()
    if image is not None:
        intensity = process_image(image)
        plot_spectrum(intensity)

if __name__ == "__main__":
    main()
