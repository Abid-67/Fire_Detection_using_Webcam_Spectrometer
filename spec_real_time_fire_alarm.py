# Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pygame

# Function to capture an image from the webcam
def capture_image(cap):
    ret, frame = cap.read()
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

# Function to calibrate the spectrum using known sources
def calibrate_spectrum(intensity, known_peaks, known_wavelengths):
    # Fit a polynomial to the known peaks and wavelengths
    calibration_fit = np.polyfit(known_peaks, known_wavelengths, 1)
    
    return calibration_fit

# Function to plot the calibrated spectrum
def plot_spectrum(intensity, calibration_fit):
    # Convert pixel positions to wavelengths using the calibration fit
    pixel_positions = np.arange(len(intensity))
    wavelengths = np.polyval(calibration_fit, pixel_positions)
    
    plt.clf()  # Clear the current figure
    plt.plot(wavelengths, intensity)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Spectrum')
    plt.pause(0.1)  # Pause to update the plot
    
    return wavelengths, intensity

# Function to compare real-time spectrum with stored spectrum
def compare_spectra(real_time_intensity, stored_intensity):
    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(real_time_intensity, stored_intensity)
    
    # Display 'fire' if the coefficient is greater than the threshold 0.8
    if correlation > 0.8:
        display_fire()
        return True
    return False

# Function to display 'fire' in a new window and play warning sound
def display_fire():
    cv2.namedWindow("Alert", cv2.WINDOW_NORMAL)
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(img, 'FIRE', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
    cv2.imshow('Alert', img)

    # Play warning sound
    pygame.mixer.init()
    pygame.mixer.music.load('fire_alarm.mp3')  # Update with the path to your warning sound file
    pygame.mixer.music.play()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to capture, process, calibrate, and plot the spectrum in real time
def main():
    # Known peak pixel positions and corresponding known wavelengths of the sources used for calibration
    # Pixel positions were found by running spec_intVSpix.py file with known sources
    known_peaks = [290, 452, 470]  # Update with your known peak pixel positions
    known_wavelengths = [440, 630, 650]  # Update with your known wavelengths in nm
    
    # Load stored spectrum from text file
    stored_data = np.loadtxt('spectrum_reference.txt', skiprows=1)
    stored_wavelengths = stored_data[:, 0]
    stored_intensity = stored_data[:, 1]

    cap = cv2.VideoCapture(0)  # change the parameter (0) to access particular webcam (e.g. 0, 1, ..) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width to 640 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height to 480 pixels

    plt.ion()  # Turn on interactive mode for real-time plotting

    try:
        while True:
            image = capture_image(cap)
            if image is not None:
                intensity = process_image(image)
                calibration_fit = calibrate_spectrum(intensity, known_peaks, known_wavelengths)
                wavelengths, real_time_intensity = plot_spectrum(intensity, calibration_fit)
                
                # Compare the real-time spectrum with the stored spectrum
                if compare_spectra(real_time_intensity, stored_intensity):
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Real-time monitoring stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()  # Turn off interactive mode
        plt.close()  # Close the plot window

if __name__ == "__main__":
    main()
