"""
Old utility functions preserved for reference.
These were replaced by the more efficient extract_all_text_from_image() approach.
"""

import os
import json
import warnings
from typing import List, Dict, Tuple
import numpy as np
import cv2
from PIL import Image

try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False


# --- Old OCR helpers (Tesseract-based) ---
def ocr_crop(crop_bgr: np.ndarray) -> str:
    """OLD: Tesseract-based OCR on a single crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Upscale and enhance
    up = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    up = clahe.apply(up)
    # Try both polarities
    def run_tess(img: np.ndarray, config: str) -> str:
        try:
            import pytesseract
            return pytesseract.image_to_string(img, config=config).strip()
        except Exception:
            return ""
    texts = []
    for inv in (False, True):
        proc = cv2.bitwise_not(up) if inv else up
        # Binarize then slight dilation to connect strokes
        bin_img = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Try multiple PSM modes
        configs = [
            '--oem 1 --psm 7',   # single line
            '--oem 1 --psm 6',   # block of text
            '--oem 1 --psm 8',   # word
        ]
        for cfg in configs:
            txt = run_tess(bin_img, cfg)
            txt = txt.replace('\n', ' ')
            txt = ' '.join(txt.split())
            if txt:
                texts.append(txt)
    # Choose the longest non-empty as best
    best = max(texts, key=len) if texts else ""
    return best


def _limit_words(text: str, max_words: int = 5) -> str:
    """Normalize whitespace and limit to a few words."""
    if not text:
        return ""
    t = text.replace("\n", " ")
    t = " ".join(t.split())
    words = t.split(" ")
    return " ".join(words[:max_words])


def llm_ocr_crop(crop_bgr: np.ndarray) -> str:
    """OLD: Use Gemini to read short text from a cropped node image.
    Returns a concise string (<=5 words). Falls back to pytesseract when LLM unavailable.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not HAS_GEMINI or not api_key:
        print("[warn] Gemini not available; using Tesseract OCR fallback.")
        return _limit_words(ocr_crop(crop_bgr))

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        # Convert crop to PIL
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        prompt = (
            "You are given a cropped image of a single flowchart element containing HANDWRITTEN text. "
            "The handwriting may vary in style, neatness, slant, and legibility - interpret it as accurately as possible. "
            "Extract ONLY the text content inside the shape (ignore shape borders, arrows, or lines). "
            "Return the raw text exactly as written with no quotes, no extra words, no commentary. "
            "\n\nExpected syntax patterns (use these to guide interpretation):\n"
            "- Variable declarations: 'Declare x, y = 5, z' or 'declare var1, var2'\n"
            "- Input statements: 'Input x, y, z' or 'input name, age'\n"
            "- Print statements: 'Print x + \" text \" + y' or 'print result'\n"
            "- Arithmetic: + - * / % (and shortcuts ++, --, +=, -=)\n"
            "- Comparisons: < > <= >= ==\n"
            "- Branches: TRUE, FALSE (case-insensitive)\n"
            "- Keywords like Print/Declare can be lowercase, but variables are case-sensitive.\n"
            "\nFlowchart elements typically contain 1-7 words. If unclear, provide your best interpretation."
        )
        content = [prompt, img]
        resp = model.generate_content(content)
        text = (resp.text or "").strip()
        return _limit_words(text)
    except Exception:
        return _limit_words(ocr_crop(crop_bgr))


def _ocr_node_concurrent(node: Dict, crops_dir: str) -> Tuple[int, str]:
    """OLD: Extract OCR text for a single node. Returns (node_id, text)."""
    crop_path = os.path.join(crops_dir, f"{node['id']}.png")
    crop = cv2.imread(crop_path)
    if crop is None or crop.size == 0:
        warnings.warn(f"Empty crop for node {node['id']} at {crop_path}")
        return node['id'], ""
    else:
        text = llm_ocr_crop(crop)
        return node['id'], text


def validate_and_correct_ocr(nodes: List[Dict], base_image_path: str, instructions: str = "") -> List[Dict]:
    """OLD: Use LLM to validate and correct OCR results by analyzing all nodes together with the image.
    This helps catch common handwriting misreadings like 'n' → 'h', 'l' → '1', etc.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not HAS_GEMINI or not api_key:
        print("[warn] Gemini not available; skipping OCR validation.")
        return nodes

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Prepare simplified nodes for validation
        validation_nodes = [{
            "id": n["id"],
            "label": n.get("label", ""),
            "text": n.get("text", ""),
            "connecting_to": n.get("connecting_to", [])
        } for n in nodes]
        
        img = Image.open(base_image_path).convert("RGB")
        
        # Build prompt with optional instructions context
        instructions_context = ""
        if instructions:
            instructions_context = (
                "\n\nCONTEXT - ASSIGNMENT INSTRUCTIONS:\n"
                "The student was asked to create a flowchart for the following task:\n"
                f"{instructions}\n\n"
                "Use these instructions as context to understand what the student is trying to accomplish. "
                "For example, if the instructions mention 'divide by 6' and you see 'x = x 1 6', "
                "the '1' is likely a misread division symbol '/', so it should be 'x = x / 6'.\n"
            )
        
        prompt = (
            "You are validating OCR text extraction from a handwritten flowchart. "
            "Given the flowchart image and the extracted text for each node, identify and correct any OCR errors.\n\n"
            "Common handwriting OCR errors to watch for:\n"
            "- Division symbol '/' misread as '1' or 'l'\n"
            "- Multiplication '*' misread as '+' or 'x'\n"
            "- 'n' misread as 'h' (or vice versa)\n"
            "- 'l' (lowercase L) misread as '1' or 'I'\n"
            "- 'o' misread as '0'\n"
            "- 'rn' misread as 'm'\n"
            "- Missing or extra spaces\n"
            "- Case confusion (lowercase/uppercase)\n\n"
            "Expected flowchart syntax patterns:\n"
            "- Variable declarations: 'Declare x, y = 5' (not 'Declare h')\n"
            "- Input: 'Input x, name' (variables should be consistent)\n"
            "- Print: 'Print result' or 'Print x + \"text\"'\n"
            "- Arithmetic: x + 1, count++, sum -= n, x / 6, x * 2\n"
            "- Comparisons: x < 10, age >= 18, count == 0\n"
            "- Branch labels: TRUE, FALSE\n"
            f"{instructions_context}\n"
            "Look at the flowchart image and the extracted texts. For each node:\n"
            "1. Check if the text makes logical sense for its position in the flow\n"
            "2. Check if variable names are consistent across connected nodes\n"
            "3. Check if syntax follows expected patterns\n"
            "4. Use the assignment instructions (if provided) to understand intended operations\n"
            "5. Correct any obvious OCR errors\n\n"
            "Return the SAME JSON array with corrected 'text' fields. Keep id, label, and connecting_to unchanged.\n"
            "Return ONLY valid JSON, no other text.\n\n"
            f"NODES TO VALIDATE:\n{json.dumps(validation_nodes, indent=2)}"
        )
        
        content = [prompt, img]
        resp = model.generate_content(content)
        text = (resp.text or "").strip()
        
        # Try to parse corrected nodes
        try:
            corrected = json.loads(text)
            if isinstance(corrected, list) and len(corrected) == len(nodes):
                # Update original nodes with corrected text
                for orig, corr in zip(nodes, corrected):
                    if isinstance(corr, dict) and "text" in corr:
                        old_text = orig.get("text", "")
                        new_text = corr["text"]
                        if old_text != new_text:
                            print(f"[correction] node {orig['id']}: '{old_text}' → '{new_text}'")
                        orig["text"] = new_text
                return nodes
        except Exception:
            # Try to extract JSON block
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    corrected = json.loads(text[start:end + 1])
                    if isinstance(corrected, list) and len(corrected) == len(nodes):
                        for orig, corr in zip(nodes, corrected):
                            if isinstance(corr, dict) and "text" in corr:
                                old_text = orig.get("text", "")
                                new_text = corr["text"]
                                if old_text != new_text:
                                    print(f"[correction] node {orig['id']}: '{old_text}' → '{new_text}'")
                                orig["text"] = new_text
                        return nodes
                except Exception:
                    pass
        
        print(f"[warn] Could not parse validation response, keeping original OCR")
        return nodes
        
    except Exception as e:
        print(f"[warn] OCR validation failed: {e}, keeping original OCR")
        return nodes
