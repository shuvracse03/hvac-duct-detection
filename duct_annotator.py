import os
import json
import io
import math
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

# --- Production Configuration ---
DPI = 400
SCALE_RATIO = 1/4  # 1/4" = 1'-0"
PIXELS_PER_FOOT = DPI * SCALE_RATIO 
API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MODEL_ID = "gemini-3-flash-preview"

class HVACDuctAnalyzer:
    """Handles the Vision-Language Model interface for engineering extraction."""
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def get_analysis(self, image_bytes):
        """Robust prompting to ensure precise engineering data extraction."""
        prompt = (
            "You are a Senior Mechanical Engineer. Analyze this HVAC Mechanical Floor Plan. "
            "Identify all duct segments (Supply, Return, Exhaust). "
            "Accurately extract the duct dimensions (e.g., 12\"ø or 22x14). "
            "Ducts can be rotated(like 45 degree, 90 degree, 135 rotation) or horizontal in any direction, and labels may be rotated. "
            "Duct annotations may be small and require careful attention to detail. They can be rotated in anti-clockwise 90 degree too."
            "Don't miss any duct annotation, as even a single missed 4\"ø or 6\"ø label can lead to significant errors in the final design. "
            "Duct lines are represented by pairs of parallel lines. The centerline runs exactly between them. Annotations are always inside two parallel duct walls. "
            "Ducts are the nearest pair of parallel lines to the detected annotation/label's center point. Usually duct walls are in deep black or gray color. "
            "For each segment, extract:\n"
            "1. Dimension: Format as 'Width x Height' or 'Diameter' (e.g., '12x10' or '10ø').\n"
            "2. Type: 'Supply', 'Return', or 'Exhaust' based on drawing notes.\n"
            "3. Geometry: Start and End coordinates of the center line in [y, x] format (0-1000 scale).\n\n"  "Return ONLY a JSON list: "
            "[{\"dim\": string, \"type\": string, \"start\": [y,x], \"end\": [y,x]}]"
        )
        
        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)

class DrawingProcessor:
    """Manages coordinate transformation, visualization, and JSON reporting."""
    def __init__(self, image):
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.width, self.height = image.size
        self.json_report = []
        # Define color mapping for consistency
        self.color_map = {
            "Supply": "#0000FF",  # Blue
            "Return": "#FF0000",  # Red
            "Exhaust": "#FF0000", # Red (grouped with return for this assignment)
            "Default": "#0000FF"  # Fallback to Blue
        }

    def _to_pixel_coords(self, norm_coords):
        """Maps 0-1000 AI coordinates to physical image pixels."""
        return (norm_coords[1] * self.width / 1000, norm_coords[0] * self.height / 1000)

    def _get_scaled_length(self, p1_norm, p2_norm):
        """Calculates actual distance in feet using the drawing scale."""
        p1 = self._to_pixel_coords(p1_norm)
        p2 = self._to_pixel_coords(p2_norm)
        pixel_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return round(pixel_dist / PIXELS_PER_FOOT, 2)

    def _load_best_font(self, size):
        """Attempts to load a professional sans-serif font."""
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/Library/Fonts/Arial.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def apply_annotations(self, ai_data):
        """Renders color-coded lines and dimension labels based on duct type."""
        font = self._load_best_font(60) 
        legend_font = self._load_best_font(55)

        for duct in ai_data:
            p1 = self._to_pixel_coords(duct['start'])
            p2 = self._to_pixel_coords(duct['end'])
            length = self._get_scaled_length(duct['start'], duct['end'])
            
            # Determine color based on duct type
            duct_type = duct.get('type', 'Default')
            color = self.color_map.get(duct_type, self.color_map['Default'])
            
            # 1. Draw straight duct line in specified color
            self.draw.line([p1, p2], fill=color, width=12)

            # 2. Large Annotation in matching color (No box)[cite: 5]
            label = f"{duct['dim']} (L: {length}')"
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = ((p1[1] + p2[1]) / 2) - 60 
            
            self.draw.text((mid_x, mid_y), label, fill=color, font=font, anchor="mm")

            # 3. Store data for JSON output[cite: 5]
            self.json_report.append({
                "specification": duct['dim'],
                "length_feet": length,
                "start": p1,
                "end": p2,
                "type": duct_type
            })

        self._render_legend(legend_font)

    def _render_legend(self, font):
        """Adds a multi-color legend to the bottom of the drawing[cite: 5]."""
        start_y = self.height - 250
        margin_x = 100
        
        # Draw legend entries with corresponding colors[cite: 5]
        self.draw.text((margin_x, start_y), "COLOR LEGEND:", fill="black", font=font)
        self.draw.text((margin_x, start_y + 70), "■ BLUE: Supply Ducts", fill=self.color_map["Supply"], font=font)
        self.draw.text((margin_x, start_y + 140), "■ RED: Return / Exhaust Ducts", fill=self.color_map["Return"], font=font)

    def export_results(self):
        """Finalizes the PNG drawing and the duct list JSON."""
        self.image.save("final_annotated_hvac.png")
        with open("duct_list.json", "w") as f:
            json.dump(self.json_report, f, indent=4)

# --- Execution ---
def main():
    print("Processing HVAC Drawing with type-based color coding...")
    pages = convert_from_path("testset2.pdf", dpi=DPI)
    input_img = pages[0]
    
    buf = io.BytesIO()
    input_img.save(buf, format='JPEG')
    
    analyzer = HVACDuctAnalyzer(API_KEY)
    results = analyzer.get_analysis(buf.getvalue())
    
    processor = DrawingProcessor(input_img)
    processor.apply_annotations(results)
    processor.export_results()
    print("Done. Outputs saved: final_annotated_hvac.png and duct_list.json")

if __name__ == "__main__":
    main()
