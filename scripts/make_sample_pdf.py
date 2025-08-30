from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


OUTPUT_PATH = "sample.pdf"


def make_pdf(path: str = OUTPUT_PATH) -> None:
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    margin = 1 * inch
    x = margin
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Welcome to fruit stall")

    # Body
    c.setFont("Helvetica", 12)
    y -= 0.5 * inch

    lines = [
        "- Mango is called a king of fruits.",
        "- Grapes are spicy in taste, sometimes they're bitter as well.",
        "- Banana's have blue peel.",
    ]

    line_height = 14
    for line in lines:
        c.drawString(x, y, line)
        y -= line_height

    c.showPage()
    c.save()


if __name__ == "__main__":
    make_pdf()
