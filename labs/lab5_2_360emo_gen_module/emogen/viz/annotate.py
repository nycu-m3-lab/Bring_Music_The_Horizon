from PIL import ImageDraw, ImageFont


def annotate_av(image, arousal: float, valence: float) -> None:
    """Draw A/V annotation in-place."""
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 60)
    except Exception as e:
        print(f"Warning: Could not load font: {e}. Using default.")
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    text = f"A={arousal}, V={valence}"

    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_h = text_bbox[3] - text_bbox[1]
    except AttributeError:
        _, text_h = draw.textsize(text, font=font)

    x = 30
    y = image.height - text_h - 40
    draw.text((x, y), text, font=font, fill="white", stroke_width=4, stroke_fill="black")

