# HVAC Duct Annotation and Analysis Pipeline

## Project Overview

This project presents an advanced pipeline for the automated extraction, analysis, and visualization of HVAC (Heating, Ventilation, and Air Conditioning) duct specifications from engineering floor plans in PDF format. The primary objective is to accurately identify duct dimensions, trace their centerlines, calculate their lengths based on the drawing's scale, and generate a color-coded visual overlay and a structured JSON report.

This system was developed as a technical assignment for an AI Lead position, demonstrating a robust, end-to-end solution that progresses from traditional OCR methods to a sophisticated Vision-Language Model (VLM) approach.

## Core Features

- **Automated Duct Detection**: Identifies duct annotations (e.g., `12"ø`, `22x14`) even when small, rotated, or using non-standard CAD fonts.
- **Duct Type Classification**: Differentiates between 'Supply', 'Return', and 'Exhaust' ducts based on drawing context.
- **Centerline Tracing**: Accurately traces the geometric centerline of each duct segment.
- **Scaled Length Calculation**: Computes the real-world length of ducts in feet by applying the drawing's scale factor (`1/4" = 1'-0"`).
- **Color-Coded Visualization**: Generates a high-resolution PNG image (`final_annotated_hvac.png`) with duct centerlines color-coded by type (Blue for Supply, Red for Return/Exhaust) for immediate visual verification.
- **Structured Data Export**: Produces a detailed `duct_list.json` file containing specifications, calculated lengths, and coordinates for each detected duct, ready for further analysis or integration into other systems.

## The Technical Journey: From OCR to VLM

The path to a successful solution involved exploring multiple computer vision and AI techniques, each with its own set of challenges and learnings.

### 1. Initial Exploration: Tesseract and EasyOCR

The initial approach involved using popular, open-source OCR libraries to perform text detection on the rasterized PDF images.

- **Tesseract OCR**: As a widely-used standard, Tesseract was the first choice. However, it struggled significantly with the specialized, vector-based fonts common in CAD drawings. It failed to recognize most duct annotations, especially the phi (`ø`) symbol and rotated text.
- **EasyOCR**: This library showed a slight improvement over Tesseract but was still unreliable. It had difficulty segmenting and correctly interpreting the compact and often rotated duct labels, leading to an unacceptably low detection rate.

**Conclusion**: Standard OCR tools are not well-suited for the nuances of CAD-generated text and symbology, making them impractical for this engineering application.

### 2. An Improved Attempt: PaddleOCR

Recognizing the limitations of the initial tools, the project pivoted to **PaddleOCR**, a more powerful OCR framework known for better performance on complex layouts and languages.

- **Advantages**: PaddleOCR demonstrated a significant leap in performance. It successfully detected a much higher percentage of the duct annotations, including those that were small or used unusual fonts.
- **Challenges**: While text detection was better, this approach was still brittle and complex. It required a multi-step, fragmented pipeline:
    1.  Run PaddleOCR to get text and bounding boxes.
    2.  Write complex regex patterns to filter for duct-specific labels.
    3.  Separately, use traditional computer vision techniques (Canny edge detection followed by Hough Line Transform) to detect all straight lines in the drawing.
    4.  Implement geometric algorithms to associate the detected text labels with the nearest detected lines.

**Conclusion**: This hybrid approach was a viable proof-of-concept but lacked robustness. The dependency on separate line detection and the heuristics for matching labels to lines made it prone to errors and difficult to scale.

### 3. The Final Solution: A Vision-Language Model (Gemini API)

The most effective and elegant solution was achieved by leveraging a powerful, multi-modal Vision-Language Model (VLM) – **Google's Gemini API**. This approach replaced the entire fragmented pipeline with a single, intelligent query.

- **The Power of a Single Prompt**: By providing the VLM with the drawing image and a carefully engineered prompt, we could instruct it to act as a "Senior Mechanical Engineer." The model was asked to perform all necessary tasks in one step:
    - Read and interpret all duct annotations.
    - Understand the spatial relationship between labels and the parallel lines forming the ducts.
    - Trace the centerline between these parallel lines.
    - Classify the duct type.
    - Return a structured JSON object with all the required information.

- **Advantages**:
    - **Accuracy**: The VLM demonstrated superior accuracy, correctly identifying even the most challenging rotated and small-font labels.
    - **Simplicity**: It consolidated a complex, multi-step process into a single API call, dramatically reducing code complexity and fragility.
    - **Robustness**: The model's inherent understanding of context and geometry eliminated the need for brittle, hand-coded heuristics.

## How to Run the Final Solution

The final, VLM-based solution is encapsulated in `duct-annotation-visualization4.py`.

**Prerequisites:**
- Python 3.x
- An API key for the Google Gemini API.

**Installation:**
```bash
pip install -r requirements.txt 
# requirements.txt should include: google-generativeai, pdf2image, Pillow
```

**Execution:**
1.  Place your PDF file (e.g., `testset2.pdf`) in the same directory.
2.  Set your `API_KEY` in the script.
3.  Run the script from the terminal:
    ```bash
    python duct-annotation-visualization4.py
    ```

**Outputs:**
- `final_annotated_hvac.png`: A high-resolution image of the floor plan with color-coded duct centerlines and labels.
- `duct_list.json`: A structured JSON file containing the detailed analysis of each duct segment.

## Future Work and Potential Enhancements

While the current Gemini-based solution is highly effective, the field of VLMs is rapidly evolving. Future iterations could explore:

- **Open-Source VLMs**: As open-source models mature, it would be beneficial to evaluate self-hosted alternatives to reduce dependency on proprietary APIs and potentially lower operational costs. Promising models include:
    - **LLaVA (Large Language and Vision Assistant)**: A leading open-source VLM that combines Vicuna (a language model) with a vision encoder.
    - **BakLLaVA**: A variant of LLaVA using the Mistral 7B model, known for its strong performance.
    - These could be deployed on-premise or in a private cloud for enhanced data privacy and control.

- **Alternative Cloud APIs**: For production environments where a managed service is preferred, other powerful VLMs could be benchmarked for performance and cost-effectiveness:
    - **Anthropic's Claude 3 (Opus/Sonnet)**: The Claude 3 model family has demonstrated state-of-the-art vision capabilities that are highly competitive with Gemini, making it an excellent candidate for evaluation.

- **Enhanced Error Correction**: Implement a validation layer where results from the VLM are cross-referenced with a more lightweight model or a simplified heuristic to flag potential inconsistencies for human review, creating a human-in-the-loop system for mission-critical applications.
