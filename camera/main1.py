#!/usr/bin/env python3
import time
import subprocess
import requests
import datetime
import io
import os
import base64
from gpiozero import Button, RotaryEncoder
from escpos.printer import Serial
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# ========== CONFIGURATION ==========

# API Settings
API_BASE_URL = "https://web-production-95a4f.up.railway.app"

# Hardware Pins
PIN_SHUTTER = 4  # The Encoder Switch (SW)
PIN_ENC_CLK = 23  # Encoder CLK
PIN_ENC_DT  = 27  # Encoder DT

# I2C Display (128x64 OLED)
I2C_PORT = 1  # GPIO 2/3 is I2C port 1 on Pi Zero 2W
I2C_ADDRESS = 0x3C  # Default I2C address for most 128x64 OLEDs
# Run 'i2cdetect -y 1' to find your display's address if 0x3C doesn't work

# Printer Settings
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600
PRINTER_WIDTH = 384

# Camera Settings
CAM_WIDTH = 1024
CAM_HEIGHT = 768

# Image Processing - Test 2 Settings
CONTRAST = 1.65  # Slightly increased for darker shadows
BRIGHTNESS = 0.85  # Reverted
SHARPNESS = 1.3
GAMMA = 0.7  # Lower gamma makes blacks darker (was 0.8)
DARKNESS = 220  # Increased for darker thermal printing
LINE_DELAY = 0.02

# Caption Settings - Matching test header size
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 20  # Larger to match test headers

# ===================================

class ThermalCam:
    def __init__(self):
        self.current_mode_index = 0
        
        # Setup OLED Display first
        try:
            serial = i2c(port=I2C_PORT, address=I2C_ADDRESS)
            self.display = ssd1306(serial)
            print("✓ OLED Display connected")
        except Exception as e:
            print(f"Display Error: {e}")
            self.display = None
        
        self.display_status("STARTING...")
        
        # Fetch modes from API
        # "normal" is a client-side mode (no API call) available online & offline
        try:
            print("Fetching modes from API...")
            self.display_status("LOADING MODES...")
            resp = requests.get(f"{API_BASE_URL}/modes", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            self.modes = ["normal"] + data["modes"]
            print(f"Modes loaded: {self.modes}")
        except Exception as e:
            print(f"API unavailable ({e}), falling back to normal mode")
            self.modes = ["normal"]
        
        # Setup Printer
        try:
            self.printer = Serial(devfile=SERIAL_PORT, baudrate=BAUD_RATE)
        except Exception as e:
            print(f"Printer Error: {e}")

        # Setup GPIO
        # bounce_time=0.1 prevents "spamming" (debouncing)
        self.shutter = Button(PIN_SHUTTER, pull_up=True, bounce_time=0.1)
        self.encoder = RotaryEncoder(PIN_ENC_CLK, PIN_ENC_DT)
        
        # Connect Events
        self.shutter.when_pressed = self.handle_shutter_press  # Show preview
        self.shutter.when_released = self.handle_shutter_release  # Take photo
        self.encoder.when_rotated = self.handle_dial
        
        print(f"System Ready. Mode: {self.modes[self.current_mode_index]}")
        self.display_mode()

    def animate_progress(self, text="LOADING"):
        """Animated progress bar for all loading states"""
        if not self.display:
            return
        
        try:
            for i in range(0, 101, 20):
                with canvas(self.display) as draw:
                    # Title
                    w = len(text) * 6
                    x = (128 - w) // 2
                    draw.text((x, 20), text, fill="white")
                    
                    # Progress bar
                    bar_width = 100
                    bar_x = (128 - bar_width) // 2
                    bar_y = 38
                    
                    # Outline
                    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + 10), outline="white", fill="black")
                    
                    # Fill
                    fill_width = int(bar_width * i / 100)
                    if fill_width > 0:
                        draw.rectangle((bar_x + 1, bar_y + 1, bar_x + fill_width - 1, bar_y + 9), fill="white")
                
                time.sleep(0.1)
        except:
            pass

    def display_status(self, status_text, large=False):
        """Display a status message on OLED"""
        if not self.display:
            return
        
        try:
            with canvas(self.display) as draw:
                if large:
                    # Large centered text for important messages
                    # Split into multiple lines if needed
                    words = status_text.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        # Rough estimate: 10 chars per line for large text
                        if len(test_line) <= 10:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    
                    # Draw centered lines
                    y_start = 32 - (len(lines) * 8)
                    for i, line in enumerate(lines):
                        # Center each line
                        w = len(line) * 6  # Rough character width
                        x = (128 - w) // 2
                        draw.text((x, y_start + i * 16), line, fill="white")
                else:
                    # Normal centered text
                    w = len(status_text) * 6  # Rough character width
                    x = (128 - w) // 2
                    draw.text((x, 28), status_text, fill="white")
        except Exception as e:
            print(f"Display update error: {e}")

    def display_mode(self):
        """Display current mode on OLED"""
        if not self.display:
            return
        
        try:
            with canvas(self.display) as draw:
                # Title
                draw.text((20, 5), "diffuji v1.0", fill="white")
                draw.line((0, 18, 128, 18), fill="white")
                
                # Current mode
                mode_name = self.modes[self.current_mode_index]
                
                # Shorten long mode names
                display_name = mode_name
                if len(display_name) > 16:
                    display_name = display_name[:16]
                
                draw.text((10, 28), "MODE:", fill="white")
                draw.text((10, 42), display_name, fill="white")
        except Exception as e:
            print(f"Display update error: {e}")

    def show_live_preview(self):
        """Show live camera preview on OLED while button is held"""
        if not self.display:
            return
        
        try:
            # Capture a quick low-res preview frame
            cmd = [
                "rpicam-still",
                "-o", "/tmp/preview.jpg",
                "--width", "256",  # Slightly higher res for better quality
                "--height", "192",
                "-t", "0",  # Immediate capture
                "--nopreview",
                "-n"  # No status display
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode != 0:
                print(f"Preview capture failed: {result.stderr}")
                return
            
            # Load and display the preview
            preview_img = Image.open("/tmp/preview.jpg")
            
            # Resize to fit OLED (128x64)
            preview_img = preview_img.resize((128, 64), Image.Resampling.LANCZOS)
            preview_img = preview_img.convert('1')  # Convert to 1-bit for OLED
            
            with canvas(self.display) as draw:
                draw.bitmap((0, 0), preview_img, fill="white")
                
        except subprocess.TimeoutExpired:
            print("Preview timeout")
        except Exception as e:
            print(f"Preview error: {e}")

    def flash_screen(self):
        """Flash the screen white briefly (camera shutter effect)"""
        if not self.display:
            return
        
        try:
            # Fill white
            with canvas(self.display) as draw:
                draw.rectangle((0, 0, 128, 64), fill="white")
            time.sleep(0.1)
            
            # Back to black
            with canvas(self.display) as draw:
                draw.rectangle((0, 0, 128, 64), fill="black")
            time.sleep(0.05)
        except Exception as e:
            print(f"Flash error: {e}")
        """Flash the screen white briefly (camera shutter effect)"""
        if not self.display:
            return
        
        try:
            # Fill white
            with canvas(self.display) as draw:
                draw.rectangle((0, 0, 128, 64), fill="white")
            time.sleep(0.1)
            
            # Back to black
            with canvas(self.display) as draw:
                draw.rectangle((0, 0, 128, 64), fill="black")
            time.sleep(0.05)
        except Exception as e:
            print(f"Flash error: {e}")

    def handle_dial(self):
        """Handle Rotary Encoder Movement"""
        self.current_mode_index = self.encoder.steps % len(self.modes)
        print(f"Mode Switched to: {self.modes[self.current_mode_index]}")
        self.display_mode()

    def get_info_string(self):
        """Get formatted date only"""
        try:
            now = datetime.datetime.now()
            
            # Check if current mode has a 4-digit year in it
            current_mode = self.modes[self.current_mode_index]
            import re
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', current_mode)
            
            if year_match:
                # Replace current year with the year from mode name
                mode_year = year_match.group(1)
                date_str = now.strftime(f"%-m/%-d/{mode_year}")
            else:
                # Use current year
                date_str = now.strftime("%-m/%-d/%Y")

            return date_str

        except Exception as e:
            return now.strftime("%Y-%m-%d")

    def take_photo(self):
        """Capture image using rpicam-still"""
        print("Capturing image...")
        
        self.animate_progress("CAPTURING")
        
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
        
        # Flash effect
        self.flash_screen()
        
        return Image.open(filename)

    def add_caption(self, img, text):
        """Adds diffuji.com at bottom-left and date at bottom-right"""
        # Convert to RGB if needed for drawing
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        
        # Font for both caption and watermark - MONOSPACE, 22pt
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
            except:
                font = ImageFont.load_default()

        watermark_text = "diffuji.com"
        
        # Get text dimensions
        bbox_date = draw.textbbox((0, 0), text, font=font)
        bbox_watermark = draw.textbbox((0, 0), watermark_text, font=font)
        
        date_w = bbox_date[2] - bbox_date[0]
        date_h = bbox_date[3] - bbox_date[1]
        watermark_w = bbox_watermark[2] - bbox_watermark[0]
        watermark_h = bbox_watermark[3] - bbox_watermark[1]
        
        margin = 8  # Small margin from edges
        outline_width = 3  # Slightly thinner outline
        
        # BOTTOM-LEFT: diffuji.com
        x_watermark = margin
        y_watermark = img.height - watermark_h - margin
        
        # Draw watermark with BLACK outline and WHITE fill
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                draw.text((x_watermark + adj_x, y_watermark + adj_y), watermark_text, font=font, fill="black")
        draw.text((x_watermark, y_watermark), watermark_text, font=font, fill="white")
        
        # BOTTOM-RIGHT: Date
        x_date = img.width - date_w - margin
        y_date = img.height - date_h - margin
        
        # Draw date with BLACK outline and WHITE fill
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                draw.text((x_date + adj_x, y_date + adj_y), text, font=font, fill="black")
        draw.text((x_date, y_date), text, font=font, fill="white")
        
        return img

    def lift_shadows(self, img):
        """Lift shadows and brighten dark areas, especially faces"""
        self.animate_progress("LIFTING")
        
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
        self.animate_progress("PROCESSING")
        
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
        """Call the DispoAPI /filter endpoint with the image and mode"""
        print(f"Calling API for mode: {mode}...")
        
        self.animate_progress("API CALL")
        
        # Save image to buffer
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)

        # POST to /filter endpoint (provider is determined server-side per mode)
        resp = requests.post(
            f"{API_BASE_URL}/filter",
            data={"mode": mode},
            files={"image": ("capture.jpg", buf, "image/jpeg")},
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()

        # Decode the returned image
        img_bytes = base64.b64decode(result["image_b64"])
        api_img = Image.open(io.BytesIO(img_bytes))

        # For search modes, text is returned instead of a transformed image
        text = result.get("text", "")
        return api_img, text

    def print_image(self, img):
        print("Printing...")
        
        self.animate_progress("PRINTING")
        
        # Set darkness if supported
        try:
            self.printer.set(density=DARKNESS)
        except:
            pass
        
        self.printer.image(img, high_density_vertical=True, high_density_horizontal=True)
        self.printer.text("\n\n\n")
        
        # Show success
        self.display_status("DONE!", large=True)
        time.sleep(1)

    def handle_shutter_press(self):
        """Show live preview when button is pressed down"""
        print("Button pressed - showing preview...")
        self.show_live_preview()

    def handle_shutter_release(self):
        current_mode = self.modes[self.current_mode_index]
        print(f"Shutter Pressed! Mode: {current_mode}")
        
        try:
            raw_img = self.take_photo()
            
            # Rotate to portrait orientation FIRST (before any processing)
            raw_img = raw_img.rotate(90, expand=True)
            
            # Lift shadows early to brighten faces and dark areas
            print("Lifting shadows...")
            raw_img = self.lift_shadows(raw_img)
            
            caption_text = self.get_info_string()
            final_img = raw_img
            api_text = ""

            # Skip API call for local-only modes
            if current_mode != "normal":
                try:
                    # Rotate 180 degrees before sending to API
                    raw_img_rotated = raw_img.rotate(180, expand=True)
                    
                    # Save image before API call
                    raw_img_rotated.save("/tmp/before_api.jpg")
                    print("Saved before_api.jpg")
                    
                    api_img, api_text = self.call_api(raw_img_rotated, current_mode)
                    
                    # Rotate 180 degrees back after API returns
                    api_img = api_img.rotate(180, expand=True)
                    final_img = api_img
                    
                    # Save image after API call (after rotating back)
                    api_img.save("/tmp/after_api.jpg")
                    print("Saved after_api.jpg")
                        
                except Exception as e:
                    print(f"API Failed: {e}")
                    self.display_status("API ERROR")
                    time.sleep(1)

            # Resize to printer width
            aspect_ratio = final_img.height / final_img.width
            new_height = int(PRINTER_WIDTH * aspect_ratio)
            final_img = final_img.resize((PRINTER_WIDTH, new_height), Image.Resampling.LANCZOS)
            
            # Rotate 180 degrees for printing
            final_img = final_img.rotate(180, expand=True)

            # Resize again after rotation
            aspect_ratio = final_img.height / final_img.width
            new_height = int(PRINTER_WIDTH * aspect_ratio)
            final_img_resized = final_img.resize((PRINTER_WIDTH, new_height), Image.Resampling.LANCZOS)

            # Add caption and watermark AFTER rotation (use API text if available)
            if api_text:
                caption_text = api_text
            final_img_with_text = self.add_caption(final_img_resized, caption_text)

            # Dither and Print (using Test 2 settings)
            dithered_img = self.process_for_thermal(final_img_with_text)
            self.print_image(dithered_img)
            
            # Return to mode display
            self.display_mode()
            print("Done.")

        except Exception as e:
            print(f"Error in capture process: {e}")
            self.display_status("ERROR!")
            time.sleep(2)
            self.display_mode()

if __name__ == "__main__":
    cam = ThermalCam()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
