import os,io
import uuid
import math
from bedrock_inv.log import get_logger
from bedrock_inv.aws_api import get_bedrock_client, invoke_embedding_model
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import fitz
from docling.document_converter import DocumentConverter

load_dotenv()
client = get_bedrock_client()

logger = get_logger(__name__)


def create_collage_with_tags(image_paths, output_path, padding=10, bg_color=(255, 255, 255), font_size=20):
    if not image_paths:
        return None

    images = [Image.open(p).convert("RGB") for p in image_paths]
    img_width, img_height = images[0].size
    images = [img.resize((img_width, img_height)) for img in images]

    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    collage_width = cols * img_width + (cols + 1) * padding
    collage_height = rows * img_height + (rows + 1) * padding

    collage = Image.new("RGB", (collage_width, collage_height), color=bg_color)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for idx, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        label = f"Image {idx + 1}"
        draw.rectangle([5, 5, 5 + font_size * len(label) // 1.8, 5 + font_size + 4], fill=(255, 255, 255))
        draw.text((8, 8), label, fill=(0, 0, 0), font=font)

        row = idx // cols
        col = idx % cols
        x = padding + col * (img_width + padding)
        y = padding + row * (img_height + padding)
        collage.paste(img, (x, y))

    collage.save(output_path)
    return output_path


def docling_pdf_locally(pdf_path, output_folder):
    doc_name = os.path.basename(pdf_path)
    doc_id = str(uuid.uuid4())

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    pdf_doc = fitz.open(pdf_path)

    picture_items = list(doc.pictures)
    page_data = {}
    prev_page_index = -1
    picture_counter = 1
    images_by_page = {}

    try:
        os.makedirs(output_folder, exist_ok=True)

        for pic_item in picture_items:
            prov = pic_item.prov[0]
            page_index = prov.page_no - 1
            bbox = prov.bbox

            if page_index != prev_page_index:
                picture_counter = 1
                prev_page_index = page_index

            x0, y_bottom, x1, y_top = bbox.l, bbox.b, bbox.r, bbox.t
            page = pdf_doc.load_page(page_index)
            page_height = page.mediabox_size.y
            fitz_bbox = (x0, page_height - y_top, x1, page_height - y_bottom)

            pixmap = page.get_pixmap(clip=fitz.Rect(fitz_bbox), dpi=300)
            output_filename = f"Page{page_index+1}_picture_{picture_counter}.png"
            output_path = os.path.join(output_folder, output_filename)
            pixmap.save(output_path)

            page_number = page_index + 1
            if page_number not in images_by_page:
                images_by_page[page_number] = []

            images_by_page[page_number].append(output_path)
            picture_counter += 1

    except Exception as e:
        logger.error(f"Error occurred while extracting images: {e}")
        return {}

    # Create collages per page
    collaged_images_by_page = {}
    for page_num, img_paths in images_by_page.items():
        collage_output = "text_and_image_chunks".join(output_folder, f"Page{page_num}_collage.png")
        collage_path = create_collage_with_tags(img_paths, collage_output)
        if collage_path:
            collaged_images_by_page[page_num] = collage_path

    # Build final page_data
    page_data = {}
    for i in range(1, len(doc.pages) + 1):
        logger.info(f"Exporting page {i} to markdown...")
        page_content = doc.export_to_markdown(page_no=i)
        logger.info(f"Embedding page {i} content...")

        embeddings = invoke_embedding_model(client, str(page_content)) if client else None

        page_data[i] = {
            'doc_id': doc_id,
            'doc_name': doc_name,
            'page_number': i,
            'text': page_content,
            'image_path': [collaged_images_by_page.get(i)] if i in collaged_images_by_page else [],
            'embedding': embeddings
        }

    return page_data
