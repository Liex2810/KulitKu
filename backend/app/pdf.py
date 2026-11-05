from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,KeepTogether

def build_pdf(diary,scans,photos=None):
    stream=BytesIO();styles=getSampleStyleSheet();story=[]
    def para(value,style="BodyText"):
        return Paragraph(escape(str(value)).replace("\n","<br/>"),styles[style])
    story += [para("KulitKu — Diary Report","Title"),para(diary["title"],"Heading1"),para(diary.get("notes", "")),Spacer(1,16)]
    story += [para("Research screening only. Model outputs are not a medical diagnosis. Normal does not exclude all skin conditions."),Spacer(1,16)]
    for scan in scans:
        block=[para(scan["created_at"],"Heading2"),para(scan.get("predicted_class") or scan["status"]),para("Model: "+scan["model_version"])]
        content=(photos or {}).get(scan["id"])
        if content:
            reader=ImageReader(BytesIO(content));width,height=reader.getSize();scale=min(360/width,240/height)
            block.append(Image(BytesIO(content),width=width*scale,height=height*scale))
        else:block.append(para("Photo not stored or unavailable."))
        block.append(Spacer(1,12));story.append(KeepTogether(block))
    if not scans:story.append(para("No linked scans."))
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.drawString(40,25,f"KulitKu | Page {doc.page}")
    SimpleDocTemplate(stream,title="KulitKu Diary",author="KulitKu",leftMargin=40,rightMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
