import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import math
import json
import hashlib
import io
import re
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

# --- Config ---
DPI = 300
API_KEY = "YOUR-GEMINI-API-KEY-HERE"
MODEL_ID = "gemini-3-flash-preview"
CACHE_FILE = "vlm_cache.json"

def get_line_params(p1, p2):
    """Calculates angle and midpoint for collinearity checks[cite: 1]."""
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return angle % np.pi, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def clean_and_group_lines(candidates, dist_thresh=110, angle_thresh=np.pi/45):
    """Merges redundant segments into single continuous centerlines[cite: 1]."""
    if not candidates: return []
    candidates.sort(key=lambda x: math.sqrt((x[1][0]-x[0][0])**2 + (x[1][1]-x[0][1])**2), reverse=True)
    
    merged = []
    used = [False] * len(candidates)
    for i in range(len(candidates)):
        if used[i]: continue
        group = [candidates[i]]
        used[i] = True
        ref_ang, ref_mid = get_line_params(candidates[i][0], candidates[i][1])
        
        for j in range(i + 1, len(candidates)):
            if used[j]: continue
            test_ang, test_mid = get_line_params(candidates[j][0], candidates[j][1])
            if abs(ref_ang - test_ang) < angle_thresh:
                d = math.sqrt((ref_mid[0] - test_mid[0])**2 + (ref_mid[1] - test_mid[1])**2)
                if d < dist_thresh:
                    cross_dist = abs((test_mid[0] - ref_mid[0]) * math.sin(ref_ang) - 
                                     (test_mid[1] - ref_mid[1]) * math.cos(ref_ang))
                    if cross_dist < 8:
                        group.append(candidates[j])
                        used[j] = True

        pts = np.array([p for line in group for p in line], dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()
        projections = [(p[0] - x0) * vx + (p[1] - y0) * vy for p in pts]
        min_p = [int(x0 + min(projections) * vx), int(y0 + min(projections) * vy)]
        max_p = [int(x0 + max(projections) * vx), int(y0 + max(projections) * vy)]
        merged.append([min_p, max_p])
    return merged

class HVACOmniTool:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.cache = self._load_cache()

    def _load_cache(self):
        """Loads analysis results to prevent redundant API calls[cite: 2]."""
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                try: return json.load(f)
                except: return {}
        return {}

    def get_vlm_analysis(self, image_bytes):
        """Extracts scale and specific duct dimensions with error handling[cite: 2]."""
        img_hash = hashlib.sha256(image_bytes).hexdigest()
        if img_hash in self.cache: return self.cache[img_hash]

        prompt = (
            "Analyze this HVAC Plan: 1. Extract scale (e.g. 1/4\"=1'-0\")[cite: 3]. "
            "2. Identify duct dimension annotations (e.g., 12x10, 18\" phi) and their locations[cite: 2]. "
            "Return JSON format: {\"scale\": \"\", \"labels\": [{\"text\":\"\", \"type\":\"Supply/Return/Exhaust\",  \"pos\":[y,x]}]}"
        )
        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        result = json.loads(response.text)
        self.cache[img_hash] = result
        with open(CACHE_FILE, 'w') as f: json.dump(self.cache, f, indent=4)
        return result

def process_hvac_assignment(pdf_path):
    images = convert_from_path(pdf_path, dpi=DPI)
    pil_img = images[0]
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Background Cleaning (Logic from line_finder.py)[cite: 1]
    xp, fp = [0, 45, 65, 255], [0, 0, 255, 255]
    lut = np.interp(np.arange(256), xp, fp).astype('uint8')
    img_contrast = cv2.LUT(gray, lut)
    _, binary = cv2.threshold(img_contrast, 220, 255, cv2.THRESH_BINARY_INV)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (int(w*0.04), int(h*0.04)), (int(w*0.84), int(h*0.65)), 255, -1)
    binary_roi = cv2.bitwise_and(binary, mask)

    lines = cv2.HoughLinesP(binary_roi, 1, np.pi/180, threshold=90, minLineLength=130, maxLineGap=15)
    candidates = []
    if lines is not None:
        for i in range(len(lines)):
            l1 = lines[i][0]
            for j in range(i + 1, len(lines)):
                l2 = lines[j][0]
                ang1, ang2 = math.atan2(l1[3]-l1[1], l1[2]-l1[0]), math.atan2(l2[3]-l2[1], l2[2]-l2[0])
                if abs(ang1 - ang2) < (np.pi / 120): 
                    m1, m2 = ((l1[0]+l1[2])/2, (l1[1]+l1[3])/2), ((l2[0]+l2[2])/2, (l2[1]+l2[3])/2)
                    dist = math.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                    if 25 < dist < 210:
                        candidates.append([[int((l1[0]+l2[0])/2), int((l1[1]+l2[1])/2)], 
                                          [int((l1[2]+l2[2])/2), int((l1[3]+l2[3])/2)]])

    final_lines = clean_and_group_lines(candidates)

    # 2. VLM Dimension Extraction (Logic from annotation_maker.py)[cite: 2]
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    tool = HVACOmniTool(API_KEY)
    vlm_data = tool.get_vlm_analysis(buf.getvalue())
    
    scale_str = vlm_data.get('scale', '1/4')
    match = re.search(r"(\d+)/(\d+)", scale_str)
    ppf = (DPI * (int(match.group(1))/int(match.group(2)))) if match else DPI*0.25

    # 3. Double Annotation Strategy[cite: 1, 2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    line_json = []

    # FIX: Use .get() to avoid KeyError if 'labels' is missing or named differently[cite: 2]
    duct_labels = vlm_data.get('labels', vlm_data.get('ducts', []))
    # 3. PIL-based Rendering for Phi Symbol Support
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_paths = [
    "C:\\Windows\\Fonts\\arial.ttf",           # Windows
    "/Library/Fonts/Arial.ttf",                # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux
    "arial.ttf"                                # Local folder fallback
    ]

    font_size = 40
    font = None

    for path in font_paths:
      if os.path.exists(path):
        font = ImageFont.truetype(path, font_size)
        break

    if font is None:
      print("Warning: TrueType font not found. Falling back to tiny default font.")
      font = ImageFont.load_default()

    for label in duct_labels:
        lx = int(label['pos'][1] * w / 1000)
        ly = int(label['pos'][0] * h / 1000)
        # Determine S/E/R suffix[cite: 2]
        d_type = label.get('type', 'Unknown')[0].upper() 
        print(label['text'])
        print('===========================')
        #clean_text = label['text'].replace("phi", "D").replace("\u03c6", "D")
        display_text = f"{label['text']} ({d_type})"

        draw.text((lx, ly-40), display_text, font=font, fill=(255, 0, 0))

    # Convert back to CV for line drawing[cite: 1]
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    line_json = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]

    # Annotation: Line IDs and Lengths in format ID(Length)[cite: 1]
    for idx, line in enumerate(final_lines):
        line_id = idx + 1
        px_len = math.sqrt((line[1][0]-line[0][0])**2 + (line[1][1]-line[0][1])**2)
        if px_len < 110: continue
        
        ft_len = round(px_len / ppf, 1)
        color = colors[idx % len(colors)]
        
        cv2.line(img_cv, tuple(line[0]), tuple(line[1]), color, 8)
        
        mid_x, mid_y = (line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2
        cv2.putText(img_cv, f"{line_id}", (mid_x, mid_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        line_json.append({"id": line_id, "length": f"{ft_len}'", "coords": line})

    cv2.imwrite("final_assignment_output.png", img_cv)
    with open("duct_lines.json", "w") as f:
        json.dump(line_json, f, indent=4)

if __name__ == "__main__":
    process_hvac_assignment("testset2.pdf")
