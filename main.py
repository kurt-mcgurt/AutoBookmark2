import logging
import sys
import datetime
import os
import shutil
import subprocess
import uuid
import glob
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import time
from PIL import Image
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
import gradio as gr


# Helper: delete files older than N days
def cleanup_old_files(directory: str, days: int = 7):
    cutoff = time.time() - days * 86400
    for f in Path(directory).glob("*"):
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                logging.info(f"Removed old file: {f}")
            except Exception as e:
                logging.error(f"Failed to remove {f}: {e}")


# Logging Setup
# ────────────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
logging_filename = f'logs/SheetExtractor_{timestamp}.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
fh = logging.FileHandler(logging_filename, mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
logging.info(f"Logging initialized. Writing to {logging_filename}")


# ────────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────────────────────────────────────
class PageDetail(BaseModel):
    page_number: int = Field(..., description="Page number")
    sheet_number: str = Field(..., description="Sheet number")
    sheet_title: str = Field(..., description="Sheet title")


class ExtractedData(BaseModel):
    total_num_pages_all_parts: int = Field(..., description="Total pages")
    pages: List[PageDetail] = Field(..., description="List of page details")


# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
class AppConfig:
    # Local output directory
    BASE_OUTPUT_DIR: str = "./outputs"
    # Must set this in Replit Secrets (Environment Variables)
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-2.5-pro-exp-03-25"
    GEMINI_TEMPERATURE: float = 0.0
    GS_RESOLUTION: int = 120
    GS_IMAGE_FORMAT: str = "pnggray"
    GEMINI_SYSTEM_PROMPT_TEMPLATE: str = """<ROLE>

**You are an expert at finding and extracting** a `sheet_number` (aka sheet no. or plan no., etc.) and a `sheet_title` **from within the 'Title Block' of each page** (each page being an image) of a construction drawings/plans document.
    * You must first identify the 'Title Block' area.
    * DO NOT use data from the center of the page. **This is not the 'Title Block'**.

</ROLE>

<CONTEXT>

* You are processing **exactly {actual_total_pages} pages of a construction drawings document** that has been converted to {actual_total_pages} single-page images.
* Each image has already been cropped to show only the **right-most 30% of the page**, containing the area where the title block is typically found.
* You will NOT see the full page—only this right-side region.
* Note: While the 'Title Block' contains the required information, its exact visual layout, positioning of elements, or surrounding
text might vary in different parts of the document set. As a hypothetical example:
    * Pgs1-49 might use 'Title_Block_Layout_1', while Pgs 50-100 might use 'Title_Block_Layout_2', etc.
    * **Title Blocks are always found on the bottom edge or right-side edge of each page**, with **`sheet_number` and `sheet_title` generally appearing along the
right-side edge or in the lower-right corner**.
    * **DO NOT use data from the top or center of the page** even if you believe it's the `sheet_title` as this data is incorrect.
    * Focus on identifying the semantic meaning of `sheet_number` and `sheet_title` within the title block area, regardless of minor layout shifts. Including:
        * Patterns in the structure of the `sheet_number` and how it relates to the corresponding `sheet_title`.

</CONTEXT>

<TERM_DEFINITIONS>

(FYI: Terms can vary slightly in these documents!)

1. `sheet_number` (aka 'Sheet No.' or similar) is the plan or drawing number and **found in the title block** on each page (image).
    * `sheet_number` hypothetical format/structure examples:
        * 'A1.1'
        * 'M2.0S1'
        * 'M-304'
        * 'S200'
        * 'A0-CS2'
        * 'BA-101'
        * There are many other alphanumerical (including decimals and hyphens) combinations!
2. `sheet_title` (aka 'Plan Title', 'Drawing Title', or similar) is the name of each Drawing/Plan and is **found in the title block** on each page (image).
    * `sheet_title` hypothetical format/structure examples:
        * '2nd Floor Plan - Area A'
        * '1st Floor Reflected Ceiling Plan'
        * 'Mechanical Schedules'
        * 'Plan and Profile'
        * More examples below!
    * ALWAYS sanitize `sheet_title` data using a combination of the following '**SANITIZATION RULES (for `sheet_title` JSON output)**':
        1. Symbols & Punctuation:
            - '-' is the only symbol allowed in the final `sheet_title` data in the JSON output.
            - '/' and '\\' in the document, become '-' in the JSON output.
            - '&' in the document, becomes 'and' in the JSON output.
            - ',', '.', and '#' in the document, are removed in the JSON output.
            - '(', ')', '[', ']', '{{', and '}}' in the document, are removed in the JSON output while keeping the enclosed text.
        2. VERY IMPORTANT (to avoid failure):
            - **ALWAYS convert `sheet_title` to 'Title Case'**.
            - **Never output `sheet_title` in ALL CAPS**.
        3. Using pattern "*appearance in document* :: **sanitized data for JSON output**" here are some real-world sanitization examples:
            - *Detached Garage #2 & #3 Enlarged Electrical Plans* :: **Detached Garage 2 and 3 Enlarged Electrical Plans**
            - *1st, 2nd, & 3rd Floor Bldg. Plans / Notes* :: **1st 2nd and 3rd Floor Bldg Plans - Notes**
            - *Grading and SESC Plan (1 of 6)* :: **Grading and Sesc Plan 1 of 6**
            - *3rd FLOOR BUILDING PLANS* :: **3rd Floor Building Plans**
            - *1st Floor Partial Bldg. Plans - Units \\\"D8-H\\\" & \\\"T1-H\\\"* :: **1st Floor Partial Bldg Plans - Units D8-H and T1-H**

</TERM_DEFINITIONS>

<TASK>
1. You will analyze the provided pages (images) page by page (image by image), sequentially, covering all {actual_total_pages} pages.
2. For each page (image), determine the corresponding `page_number` (starting from 1 for the first image and incrementing sequentially up to {actual_total_pages} ).
3. Extract the Sheet Number and Sheet Title by visually reading the TITLE BLOCK within each page (image), disregard most if not all of the body of the pages.
4. Your output **MUST** be a single JSON **object** conforming to the specified response schema.
5. This JSON object must contain:
    * `total_num_pages_all_parts`: The total number of pages. This **MUST** equal {actual_total_pages}.
    * `pages`: A JSON array containing exactly {actual_total_pages} objects, one for each page/image processed.
        * Each object in this array must contain the `page_number`, `sheet_number`, and `sheet_title` for that specific page/image.
</TASK>

<KEYS_TO_SUCCESS>

These are your KEYS TO SUCCESS:
1. **NEVER make assumptions**
2. Every sheet number/title in your output JSON will use **REAL DATA that you visually read/extracted from the page images**
3. Do not include any bullet points, numbering, or extra commentary outside the JSON structure.
4. The `pages` array **MUST** contain exactly {actual_total_pages} entries, ordered sequentially by `page_number` starting from 1 and ending at {actual_total_pages}.
5. Ensure the value for `total_num_pages_all_parts` **MUST** be exactly {actual_total_pages}.
6. The data will be consistently located from page to page in most cases aside from cover/title sheets and oddball documents.
7. The first page (`page_number` = 1), if it's a cover/title sheet or list of drawings, will always have `sheet_number` "CS" or "TS", and `sheet_title` of "Cover Sheet" or "Title Sheet".
8. Always output "pretty-printed" human-readable JSON.

</KEYS_TO_SUCCESS>"""


# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions (unchanged except no Colab dependencies)
# ────────────────────────────────────────────────────────────────────────────────
def find_executable(tool_name: str, common_names: List[str]) -> str:
    logging.info(f"Locating {tool_name} executable...")
    for name in common_names:
        path = shutil.which(name)
        if path:
            logging.info(f"{tool_name} found at: {path}")
            return path
    raise FileNotFoundError(
        f"{tool_name} executable not found: tried {common_names}")


def get_processing_paths(output_base_dir: str,
                         original_filename: str) -> Dict[str, str]:
    safe_stem = Path(original_filename).stem.replace(" ", "_")
    unique_id = uuid.uuid4().hex[:8]
    temp_dir = Path(output_base_dir) / f"proc_{safe_stem}_{unique_id}"
    image_dir = temp_dir / "images"
    resp_json = temp_dir / f"{safe_stem}_response.json"
    bmk_file = temp_dir / f"{safe_stem}_bookmarks.txt"
    final_pdf = Path(output_base_dir) / f"Autobookmarked_{original_filename}"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    logging.info(f"Created temp dirs under {temp_dir}")
    return {
        "temp_dir": str(temp_dir),
        "image_output_dir": str(image_dir),
        "response_file_path": str(resp_json),
        "bookmark_file_path": str(bmk_file),
        "final_output_path": str(final_pdf),
    }


def pdf_to_images(pdf_path: str, output_dir: str, gs_path: str,
                  resolution: int, image_format: str) -> List[str]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    ext = "png" if "png" in image_format else "jpg"
    pattern = os.path.join(output_dir, f"page_%04d.{ext}")
    cmd = [
        gs_path, "-dNOPAUSE", "-dBATCH", "-dSAFER", "-q",
        f"-sDEVICE={image_format}", f"-r{resolution}",
        f"-sOutputFile={pattern}", pdf_path
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
    files = glob.glob(os.path.join(output_dir, f"page_*.{ext}"))
    files.sort(key=lambda f: int(re.search(r'page_(\d+)\.', f).group(1)))
    return files


def generate_bookmarks_ai(client: genai.Client, model_name: str,
                          temperature: float, system_prompt_template: str,
                          image_paths: List[str],
                          response_schema: BaseModel) -> Dict:
    parts = []
    for path in image_paths:
        with open(path, "rb") as f:
            data = f.read()
        mime = Image.MIME.get(Image.open(path).format.upper()) or "image/png"
        parts.append(genai_types.Part.from_bytes(data=data, mime_type=mime))
    prompt = system_prompt_template.format(actual_total_pages=len(parts))
    cfg = genai_types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=prompt,
    )
    resp = client.models.generate_content(model=model_name,
                                          contents=parts,
                                          config=cfg)
    raw = resp.text
    core = raw[raw.find("{"):raw.rfind("}") + 1]
    return json.loads(core)


def convert_ai_response_to_pdftk(response_data: Dict,
                                 output_bmk_path: str) -> bool:
    pages = response_data.get("pages", [])
    lines = []
    for p in pages:
        title = f"{p['sheet_number']} {p['sheet_title']}"
        lines.append(
            f"BookmarkBegin\nBookmarkTitle: {title}\nBookmarkLevel: 1\nBookmarkPageNumber: {p['page_number']}\n\n"
        )
    os.makedirs(os.path.dirname(output_bmk_path), exist_ok=True)
    with open(output_bmk_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return True


def apply_bookmarks(pdftk_path: str, input_pdf: str, bookmark_data: str,
                    output_pdf: str) -> bool:
    cmd = [
        pdftk_path, input_pdf, "update_info", bookmark_data, "output",
        output_pdf
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return os.path.exists(output_pdf) and os.path.getsize(output_pdf) > 0


def process_pdf_workflow(input_pdf: str, paths: Dict[str, str], gs_path: str,
                         pdftk_path: str, client: genai.Client,
                         config: AppConfig) -> str:
    try:
        imgs = pdf_to_images(input_pdf, paths["image_output_dir"], gs_path,
                             config.GS_RESOLUTION, config.GS_IMAGE_FORMAT)
        # crop images to rightmost 30%
        crop_dir = os.path.join(paths["temp_dir"], "cropped")
        os.makedirs(crop_dir, exist_ok=True)
        cropped = []
        for img in imgs:
            out = os.path.join(crop_dir, os.path.basename(img))
            with Image.open(img) as im:
                w, h = im.size
                im.crop((int(w * 0.8), 0, w, h)).save(out)
            cropped.append(out)
        ai_resp = generate_bookmarks_ai(client, config.GEMINI_MODEL_NAME,
                                        config.GEMINI_TEMPERATURE,
                                        config.GEMINI_SYSTEM_PROMPT_TEMPLATE,
                                        cropped, ExtractedData)
        with open(paths["response_file_path"], "w") as f:
            json.dump(ai_resp, f, indent=2)
        if not convert_ai_response_to_pdftk(ai_resp,
                                            paths["bookmark_file_path"]):
            raise RuntimeError("Bookmark conversion failed")
        if not apply_bookmarks(pdftk_path, input_pdf,
                               paths["bookmark_file_path"],
                               paths["final_output_path"]):
            raise RuntimeError("Applying bookmarks failed")
        return paths["final_output_path"]
    finally:
        # cleanup everything except final output
        shutil.rmtree(paths["temp_dir"], ignore_errors=True)


def handle_uploaded_pdf(uploaded_pdf_path: str, config: AppConfig,
                        gs_path: str, pdftk_path: str,
                        client: genai.Client) -> str:
    original = os.path.basename(uploaded_pdf_path)
    paths = get_processing_paths(config.BASE_OUTPUT_DIR, original)
    return process_pdf_workflow(uploaded_pdf_path, paths, gs_path, pdftk_path,
                                client, config)


# ────────────────────────────────────────────────────────────────────────────────
# Startup and Gradio Interface
# ────────────────────────────────────────────────────────────────────────────────
logging.info("Starting Autobookmark application...")

# Load configuration
try:
    config = AppConfig()
    if not config.GEMINI_API_KEY:
        raise ValueError("Environment variable GEMINI_API_KEY is not set.")
    logging.info("Configuration loaded.")
except Exception as e:
    logging.critical(f"Configuration error: {e}")
    sys.exit(1)

# Now that config is known, clean up old files
os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)
cleanup_old_files("logs", days=7)
cleanup_old_files(config.BASE_OUTPUT_DIR, days=7)

# Initialize Gemini client
client = genai.Client(api_key=config.GEMINI_API_KEY)
logging.info("Gemini client initialized.")

# Locate external tools
gs_names = ["gs"] if sys.platform.startswith("linux") else [
    "gswin64c", "gswin32c", "gs"
]
pdftk_names = ["pdftk"]
gs_path = find_executable("Ghostscript", gs_names)
pdftk_path = find_executable("PDFtk", pdftk_names)


# Gradio app with Blocks UI
# ────────────────────────────────────────────────────────────────────────────────
def gradio_process(uploaded_file):
    path = str(uploaded_file)
    return handle_uploaded_pdf(path, config, gs_path, pdftk_path, client)


with gr.Blocks(theme=gr.themes.Default(primary_hue="green")) as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>Autobookmark</h1>"
        "<p style='text-align: center;'>Upload a PDF and get it back with Gemini-powered bookmarks.</p>"
    )
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload your PDF",
                                file_types=[".pdf"],
                                type="filepath")
            submit_btn = gr.Button("Submit", elem_id="submit-btn")
        with gr.Column():
            download_btn = gr.DownloadButton(
                label="⬇️ Download Bookmarked PDF",
                value=None,
                visible=False,
                elem_id="download-btn",
                scale=2)

    submit_btn.click(
        lambda file: gr.update(visible=True, value=gradio_process(file)),
        inputs=pdf_input,
        outputs=download_btn)

demo.launch(server_name="0.0.0.0", share=True)
