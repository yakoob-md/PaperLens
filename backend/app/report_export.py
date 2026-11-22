from io import BytesIO
from typing import Literal
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate

router = APIRouter()


class ReportBlock(BaseModel):
    kind: Literal["title", "heading", "subheading", "text", "bullet"]
    text: str = Field(max_length=100000)


class PDFExport(BaseModel):
    blocks: list[ReportBlock] = Field(min_length=1, max_length=3000)


def render_report(blocks: list[ReportBlock]) -> bytes:
    destination = BytesIO()
    ink = HexColor("#202833")
    accent = HexColor("#453990")
    styles = {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=29,
                                textColor=ink, spaceAfter=12, keepWithNext=True),
        "heading": ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=15, leading=19,
                                  textColor=accent, spaceBefore=13, spaceAfter=5, keepWithNext=True),
        "subheading": ParagraphStyle("Subheading", fontName="Helvetica-Bold", fontSize=11, leading=15,
                                     textColor=ink, spaceBefore=7, spaceAfter=3, keepWithNext=True),
        "text": ParagraphStyle("Text", fontName="Helvetica", fontSize=10, leading=14,
                               textColor=ink, alignment=TA_LEFT, spaceAfter=6),
        "bullet": ParagraphStyle("Bullet", fontName="Helvetica", fontSize=10, leading=14,
                                 textColor=ink, leftIndent=13, bulletIndent=3, spaceAfter=4),
    }
    report = SimpleDocTemplate(destination, pagesize=A4, leftMargin=48, rightMargin=48,
                               topMargin=56, bottomMargin=50, title=blocks[0].text[:200], author="PaperLens")

    def draw_page(canvas, doc):
        canvas.setFillColor(accent)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(doc.leftMargin, A4[1] - 30, "PaperLens")
        canvas.setStrokeColor(HexColor("#dfdce7"))
        canvas.line(doc.leftMargin, 38, A4[0] - doc.rightMargin, 38)
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(A4[0] - doc.rightMargin, 24, str(doc.page))

    story = [
        Paragraph(escape(block.text).replace("\n", "<br/>"), styles[block.kind],
                  bulletText="-" if block.kind == "bullet" else None)
        for block in blocks if block.text.strip()
    ]
    report.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return destination.getvalue()


@router.post("/api/reports/pdf")
def export_pdf(payload: PDFExport):
    if sum(len(block.text) for block in payload.blocks) > 600000:
        raise HTTPException(413, "This report is too large to export as PDF.")
    return Response(render_report(payload.blocks), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="paperlens-report.pdf"'})
