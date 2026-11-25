d=Document(); sec=d.sections[0]
sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.65); sec.bottom_margin=Cm(1.65); sec.left_margin=Cm(1.7); sec.right_margin=Cm(1.7)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 st=d.styles[name]; st.font.name='Times New Roman'; st.font.color.rgb=RGBColor(0,0,0)
st=d.styles['Normal']; st.font.size=Pt(10); st.paragraph_format.space_after=Pt(5); st.paragraph_format.line_spacing=1.04
for name in ['Heading 1','Heading 2']:
 d.styles[name].font.size=Pt(10.5); d.styles[name].paragraph_format.space_before=Pt(7); d.styles[name].paragraph_format.space_after=Pt(4)
d.styles['Caption'].font.size=Pt(8); d.styles['Caption'].font.italic=False
def p(text,bold=False,size=None,align=None):
 q=d.add_paragraph(); q.alignment=align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY; r=q.add_run(text); r.bold=bold
 if size:r.font.size=Pt(size)
 return q
def h(text):d.add_paragraph(text,'Heading 1')
def sub(text):d.add_paragraph(text,'Heading 2')
def table(headers,rows,caption):
 cap=p(caption,bold=True,size=8,align=WD_ALIGN_PARAGRAPH.CENTER);cap.paragraph_format.keep_with_next=True
 t=d.add_table(rows=1,cols=len(headers)); t.autofit=False
 widths=[1.65,1.7] if len(headers)==2 else [1.35,.8,1.2]
 for col,w in zip(t.columns,widths): col.width=Inches(w)
 for row in t.rows:
  for c,w in zip(row.cells,widths):c.width=Inches(w)
 props=t._tbl.tblPr
 tw=props.find(qn('w:tblW'));tw.set(qn('w:w'),str(int(sum(widths)*1440)));tw.set(qn('w:type'),'dxa')
 for c,v in zip(t.rows[0].cells,headers): c.text=v
 for row in rows:
  newrow=t.add_row()
  for c,v,w in zip(newrow.cells,row,widths):c.text=str(v);c.width=Inches(w)
 for ri,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
  for c in row.cells:
   pr=c._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    el=OxmlElement('w:'+edge); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
   pr.append(borders)
   if ri==0:
    sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'EAEAEA');pr.append(sh)
   for q in c.paragraphs:
    q.paragraph_format.space_after=Pt(3);q.paragraph_format.space_before=Pt(3);q.paragraph_format.line_spacing=1;q.paragraph_format.keep_with_next=(ri<len(t.rows)-1)
    for r in q.runs:r.font.size=Pt(8);r.bold=ri==0
 return t
def image(name,caption,width=3.35):
 q=d.add_paragraph(); q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.keep_with_next=True;q.add_run().add_picture(str(fig/name),width=Inches(width))
 q=d.add_paragraph(caption,'Caption');q.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
