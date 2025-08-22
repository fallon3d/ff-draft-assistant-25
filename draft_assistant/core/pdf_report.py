"""
PDF report generation for draft summary.
"""
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_pdf(league_name, picks, user_slot):
    """
    Generate a PDF report of the draft.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Cover page
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height-100, league_name or "Draft Report")
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height-130, f"Team Slot: {user_slot}")
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    c.drawCentredString(width/2, height-150, date_str)
    c.showPage()

    # Picks page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, "Draft Picks")
    c.setFont("Helvetica", 10)
    y = height - 80
    c.drawString(50, y, "Round")
    c.drawString(100, y, "Pick")
    c.drawString(140, y, "Team")
    c.drawString(240, y, "Player")
    y -= 20
    for pick in picks:
        if y < 50:
            c.showPage()
            y = height - 50
        rnd = str(pick.get("round"))
        pk = str(pick.get("pick_no"))
        team = pick.get("team", "Unknown")
        meta = pick.get("metadata", {})
        player = f"{meta.get('first_name', '')} {meta.get('last_name', '')}"
        player = player.strip()
        c.drawString(50, y, rnd)
        c.drawString(100, y, pk)
        c.drawString(140, y, team)
        c.drawString(240, y, player)
        y -= 15

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
