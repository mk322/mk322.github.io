# Helper definitions copied from the original figure source.

def label(t,x,y,size=30,bold=False,color=GRAPHITE,anchor='left'):
    text_mid(t,x,y,HN_B if bold else HN_M,size,color,anchor)

def symbol(spec,x,y,size=38,color=GRAPHITE):
    base,idx,sup=spec
    bw=pdfmetrics.stringWidth(base,'TimesNRI',size)
    extra=max(pdfmetrics.stringWidth(idx,'TimesNRI',size*.50),pdfmetrics.stringWidth(sup,'TimesNR',size*.50)) if idx or sup else 0
    start=x-(bw+extra)/2
    text(base,start,y+size*.30,'TimesNRI',size,color)
    if idx: text(idx,start+bw,y+size*.40,'TimesNRI',size*.50,color)
    if sup: text(sup,start+bw,y-size*.07,'TimesNR',size*.50,color)

def tok(spec,x,y,kind=None,w=82,h=51):
    k=kind or spec[0]; fill,stroke=C[k]
    rounded_rect(x-w/2,y-h/2,w,h,10,HexColor(fill),HexColor(stroke),1.8,False)
    symbol(spec,x,y,size=34 if h<50 else 38)

def arr(x,y,x2,y2): open_straight(x,y,x2,y2,SLATE,3.5,10)

def stop_marker(x,y):
    # Small stop-bar badge on the upper-right token corner; not a forward edge.
    rounded_rect(x-12,y-12,24,24,5,CARD,GRAPHITE,1.8,False)
    line(x-6,y,x+6,y,GRAPHITE,3)
