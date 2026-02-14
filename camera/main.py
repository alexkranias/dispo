#!/usr/bin/env python3
import time
import subprocess
import requests
import datetime
import io
import os
from gpiozero import Button, RotaryEncoder
from escpos.printer import Serial
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont

# ========== CONFIGURATION ==========

# Hardware Pins
PIN_SHUTTER = 22  # The Encoder Switch (SW)
PIN_ENC_CLK = 23  # Encoder CLK
PIN_ENC_DT  = 27  # Encoder DT

# Printer Settings
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600
PRINTER_WIDTH = 384

# Camera Settings
CAM_WIDTH = 1024
CAM_HEIGHT = 768

# Modes
MODES = ["DIRECT_PRINT", "API_FILTER_1", "API_FILTER_2"]

# Image Processing - Test 2 Settings
CONTRAST = 1.6
BRIGHTNESS = 0.85
SHARPNESS = 1.3
GAMMA = 0.8
DARKNESS = 200
LINE_DELAY = 0.02

# Caption Settings - Matching test header size
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 20  # Larger to match test headers

# ===================================

class ThermalCam:
    def __init__(self):
        self.current_mode_index = 0
        
        # Setup Printer
        try:
            self.printer = Serial(devfile=SERIAL_PORT, baudrate=BAUD_RATE)
        except Exception as e:
            print(f"Printer Error: {e}")

        # Setup GPIO
        # bounce_time=0.1 prevents "spamming" (debouncing)
        self.shutter = Button(PIN_SHUTTER, pull_up=True, bounce_time=0.1)
        self.encoder = RotaryEncoder(PIN_ENC_CLK, PIN_ENC_DT, max_steps=len(MODES)-1, wrap=True)
        
        # Connect Events
        self.shutter.when_released = self.handle_shutter
        self.encoder.when_rotated = self.handle_dial
        
        print(f"System Ready. Mode: {MODES[self.current_mode_index]}")

    def handle_dial(self):
        """Handle Rotary Encoder Movement"""
        step = self.encoder.steps
        self.current_mode_index = step % len(MODES)
        print(f"Mode Switched to: {MODES[self.current_mode_index]}")

    def get_info_string(self):
        """Get formatted date and location"""
        try:
            # 1. Get Date (Format: 11:59AM 1/12/2026)
            now = datetime.datetime.now()
            # %-m and %-d remove zero padding on Linux (e.g. 01 -> 1)
            date_str = now.strftime("%I:%M%p %-m/%-d/%Y") 

            # 2. Get Location (IP Based)
            loc_str = "Unknown"
            try:
                r = requests.get('http://ip-api.com/json/', timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    city = data.get('city', 'Unknown')
                    region = data.get('region', '') # State code like CA
                    loc_str = f"{city}, {region}"
            except:
                pass

            return f"{date_str} {loc_str}"

        except Exception as e:
            return now.strftime("%Y-%m-%d %H:%M")

    def take_photo(self):
        """Capture image using rpicam-still"""
        print("Capturing image...")
        filename = "/tmp/capture.jpg"
        # Using the new rpicam-still command
        cmd = [
            "rpicam-still",
            "-o", filename,
            "--width", str(CAM_WIDTH),
            "--height", str(CAM_HEIGHT),
            "-t", "100", 
            "--nopreview"
        ]
        subprocess.run(cmd, check=True)
        return Image.open(filename)

    def add_caption(self, img, text):
        """Adds text directly on image with no background band"""
        # Convert to RGB if needed for drawing
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        padding = 10
        
        # Position text at bottom
        x = (img.width - text_w) / 2
        y = img.height - text_h - padding
        
        # Draw text with black fill and white outline for readability
        outline_width = 2
        # Draw outline
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                draw.text((x + adj_x, y + adj_y), text, font=font, fill="white")
        # Draw main text
        draw.text((x, y), text, font=font, fill="black")
        
        return img

    def lift_shadows(self, img):
        """Lift shadows and brighten dark areas, especially faces"""
        # Convert to LAB color space for better shadow/highlight control
        # PIL doesn't have LAB, so we'll use RGB adjustments
        
        # Method 1: Selective brightness boost for dark areas
        pixels = img.load()
        width, height = img.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                # Calculate luminance
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                
                # If pixel is dark (shadow), boost it more
                if luminance < 128:
                    # Shadow lift factor - more boost for darker pixels
                    boost = 1.0 + (128 - luminance) / 128 * 0.6  # Up to 60% boost
                    r = min(255, int(r * boost))
                    g = min(255, int(g * boost))
                    b = min(255, int(b * boost))
                    pixels[x, y] = (r, g, b)
        
        # Method 2: Apply S-curve to preserve highlights while lifting shadows
        img = ImageEnhance.Brightness(img).enhance(1.15)
        
        # Method 3: Reduce contrast slightly to compress shadows
        img = ImageEnhance.Contrast(img).enhance(0.95)
        
        return img

    def apply_ordered_dithering(self, img):
        """Apply Ordered (Bayer) dithering - Test 2 method"""
        threshold_map = [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]
        
        img_array = img.copy()
        pixels = img_array.load()
        width, height = img_array.size
        
        for y in range(height):
            for x in range(width):
                threshold = (threshold_map[y % 4][x % 4] / 16.0) * 255
                old_pixel = int(pixels[x, y]) if isinstance(pixels[x, y], (int, float)) else pixels[x, y]
                pixels[x, y] = 255 if old_pixel > threshold else 0
        
        return img_array.convert('1')

    def process_for_thermal(self, img):
        """Resize and process for Printer using Test 2 settings"""
        aspect_ratio = img.height / img.width
        new_height = int(PRINTER_WIDTH * aspect_ratio)
        img = img.resize((PRINTER_WIDTH, new_height), Image.Resampling.LANCZOS)
        
        img = img.convert('L') 
        img = ImageOps.autocontrast(img, cutoff=2)
        img = ImageEnhance.Contrast(img).enhance(CONTRAST)
        img = ImageEnhance.Brightness(img).enhance(BRIGHTNESS)
        img = ImageEnhance.Sharpness(img).enhance(SHARPNESS)
        
        img = img.point(lambda x: int(255 * (x / 255) ** GAMMA))
        
        # Apply ordered dithering instead of default
        img = self.apply_ordered_dithering(img)
        return img

    def call_api(self, pil_image, mode):
        # Placeholder for API
        print(f"Calling API for mode: {mode}...")
        return pil_image, f"Mode: {mode}"

    def print_image(self, img):
        print("Printing...")
        
        # Set darkness if supported
        try:
            self.printer.set(density=DARKNESS)
        except:
            pass
        
        self.printer.image(img, high_density_vertical=True, high_density_horizontal=True)
        self.printer.text("\n\n\n")

    def handle_shutter(self):
        current_mode = MODES[self.current_mode_index]
        print(f"Shutter Pressed! Mode: {current_mode}")
        
        try:
            raw_img = self.take_photo()
            
            # Lift shadows early to brighten faces and dark areas
            print("Lifting shadows...")
            raw_img = self.lift_shadows(raw_img)
            
            caption_text = self.get_info_string()
            final_img = raw_img

            if current_mode != "DIRECT_PRINT":
                try:
                    api_img, api_caption = self.call_api(raw_img, current_mode)
                    final_img = api_img
                    # Update caption based on API if needed
                except Exception as e:
                    print(f"API Failed: {e}")

            # Resize First to get aspect ratio right for printing
            aspect_ratio = final_img.height / final_img.width
            new_height = int(PRINTER_WIDTH * aspect_ratio)
            final_img = final_img.resize((PRINTER_WIDTH, new_height), Image.Resampling.LANCZOS)

            # Add Caption (now overlaid on image, no white band)
            final_img_with_text = self.add_caption(final_img, caption_text)

            # Dither and Print (using Test 2 settings)
            dithered_img = self.process_for_thermal(final_img_with_text)
            self.print_image(dithered_img)
            print("Done.")

        except Exception as e:
            print(f"Error in capture process: {e}")

if __name__ == "__main__":
    cam = ThermalCam()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")