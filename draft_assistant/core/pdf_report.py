"""
Simple PDF report via ReportLab.
"""

from __future__ import annotations
from io import BytesIO
from typing import List, Optional
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

def generate_pdf(league_name: str, picks_log: List[dict], my_slot: Optional[int]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elems = []

    title = f"Draft Report — {league_name}"
    elems.append(Paragraph(title, styles["Title"]))
    if my_slot:
        elems.append(Paragraph(f"Your slot: {my_slot}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    # Pick log table
    data = [["Round", "Pick", "Team", "Player", "Pos"]]
    for p in picks_log or []:
        meta = p.get("metadata") or {}
        nm = f"{meta.get('first_name','')} {meta.get('last_name','')}".strip()
        data.append([p.get("round"), p.get("pick_no"), p.get("team",""), nm, (meta.get("position") or "")])

    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#222222")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.lightgrey]),
    ]))
    elems.append(tbl)

    doc.build(elems)
    return buf.getvalue()
