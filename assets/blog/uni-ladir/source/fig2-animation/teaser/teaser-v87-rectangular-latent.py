# Helper definitions copied from the original figure source.

def register_fonts():
    avenir = "/System/Library/Fonts/Avenir Next.ttc"
    pdfmetrics.registerFont(TTFont("Avenir", avenir, subfontIndex=7))
    pdfmetrics.registerFont(TTFont("Avenir-Medium", avenir, subfontIndex=5))
    pdfmetrics.registerFont(TTFont("Avenir-Demi", avenir, subfontIndex=2))
    pdfmetrics.registerFont(TTFont("Avenir-Bold", avenir, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Avenir-Heavy", avenir, subfontIndex=8))
    pdfmetrics.registerFont(TTFont("TimesNR", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNRI", "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNRB", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"))
    helvetica_neue = "/System/Library/Fonts/HelveticaNeue.ttc"
    pdfmetrics.registerFont(TTFont("HelveticaNeue", helvetica_neue, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("HelveticaNeue-Bold", helvetica_neue, subfontIndex=1))
    pdfmetrics.registerFont(TTFont("HelveticaNeue-Italic", helvetica_neue, subfontIndex=2))
    pdfmetrics.registerFont(TTFont("HelveticaNeue-Medium", helvetica_neue, subfontIndex=10))

def by(top_y):
    return H - top_y

def rounded_rect(x, y, w, h, r, fill, stroke=HAIR, sw=1.0, shadow=True):
    if shadow:
        c.saveState()
        c.setFillColor(Color(0.10, 0.12, 0.15, alpha=0.045))
        c.setStrokeColor(Color(0, 0, 0, alpha=0))
        c.roundRect(x + 1.8, H - y - h - 3.2, w, h, r, fill=1, stroke=0)
        c.restoreState()
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(sw)
    c.roundRect(x, H - y - h, w, h, r, fill=1, stroke=1)
    c.restoreState()

def line(x1, y1, x2, y2, color=HAIR, width=1, dash=None):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setLineCap(1)
    if dash:
        c.setDash(dash)
    c.line(x1, by(y1), x2, by(y2))
    c.restoreState()

def text(s, x, y, font="HelveticaNeue", size=16, color=GRAPHITE, anchor="left"):
    c.saveState()
    c.setFillColor(color)
    c.setFont(font, size)
    if anchor == "center":
        c.drawCentredString(x, by(y), s)
    elif anchor == "right":
        c.drawRightString(x, by(y), s)
    else:
        c.drawString(x, by(y), s)
    c.restoreState()

def text_mid(s, x, y, font="HelveticaNeue", size=16, color=GRAPHITE, anchor="center"):
    # y is optical vertical center in top-origin coordinates.
    text(s, x, y + size * 0.34, font, size, color, anchor)

def arrowhead(x, y, dx, dy, color, size=9.0):
    mag = math.hypot(dx, dy) or 1.0
    ux, uy = dx / mag, dy / mag
    px, py = -uy, ux
    bx, byy = x - ux * size, y - uy * size
    pts = [
        (x, y),
        (bx + px * size * 0.48, byy + py * size * 0.48),
        (bx - px * size * 0.48, byy - py * size * 0.48),
    ]
    p = c.beginPath()
    p.moveTo(pts[0][0], by(pts[0][1]))
    p.lineTo(pts[1][0], by(pts[1][1]))
    p.lineTo(pts[2][0], by(pts[2][1]))
    p.close()
    c.saveState()
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()

def icon_speech(cx, cy, color):
    c.saveState(); c.setStrokeColor(color); c.setLineWidth(2.0); c.setLineCap(1)
    c.circle(cx, by(cy), 22, fill=0, stroke=1)
    p = c.beginPath(); p.moveTo(cx - 15, by(cy + 16)); p.lineTo(cx - 24, by(cy + 27)); p.lineTo(cx - 7, by(cy + 21)); c.drawPath(p, fill=0, stroke=1)
    for off in (-7, 0, 7): line(cx - 10, cy + off, cx + 10, cy + off, color, 1.5)
    c.restoreState()

def icon_visual(cx, cy, color):
    c.saveState(); c.setStrokeColor(color); c.setLineWidth(2.0); c.setLineJoin(1)
    c.roundRect(cx - 27, by(cy + 22), 54, 44, 5, fill=0, stroke=1)
    c.circle(cx - 13, by(cy - 9), 4.3, fill=0, stroke=1)
    p = c.beginPath(); p.moveTo(cx - 22, by(cy + 13)); p.lineTo(cx - 8, by(cy - 1)); p.lineTo(cx + 1, by(cy + 8)); p.lineTo(cx + 18, by(cy - 8)); p.lineTo(cx + 24, by(cy - 1)); c.drawPath(p, fill=0, stroke=1)
    c.restoreState()

def icon_cube(cx, cy, color):
    c.saveState(); c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.8); c.setDash(3, 3)
    pts = [(cx,cy-23),(cx+23,cy-11),(cx+23,cy+14),(cx,cy+28),(cx-23,cy+14),(cx-23,cy-11),(cx,cy-23)]
    for a,b in zip(pts[:-1],pts[1:]): line(a[0],a[1],b[0],b[1],color,1.7,[3,3])
    line(cx,cy-23,cx,cy+28,color,1.7,[3,3]); line(cx-23,cy-11,cx,cy+2,color,1.7,[3,3]); line(cx+23,cy-11,cx,cy+2,color,1.7,[3,3])
    for px,py in pts[:-1]: c.circle(px,by(py),2.5,fill=1,stroke=0)
    c.restoreState()

def open_head(x, y, dx, dy, color, size=9, width=1.8):
    # One filled, triangular arrowhead family at every scale.
    arrowhead(x,y,dx,dy,color,size)

def open_straight(x1,y1,x2,y2,color=GRAPHITE,width=1.6,head=7):
    mag=math.hypot(x2-x1,y2-y1) or 1
    ux,uy=(x2-x1)/mag,(y2-y1)/mag
    line(x1,y1,x2-ux*head*.65,y2-uy*head*.65,color,width)
    open_head(x2,y2,x2-x1,y2-y1,color,head,width)
