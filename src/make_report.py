"""Render the versioned technical report as DOCX and plain text."""
import argparse,re
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor


def clean_inline(text):
    return text.replace('**','').replace('`','')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='.')
    args=parser.parse_args()
    root=Path(args.root)
    source=root/'docs/technical_report.md'
    lines=source.read_text(encoding='utf-8').splitlines()
    doc=Document()
    doc.core_properties.title='Регрессия индекса селективности соединений в отношении VACV'
    doc.core_properties.subject='Данные, структурная валидация, результаты и ограничения'
    doc.core_properties.author='VACV SI Project'
    doc.core_properties.last_modified_by='VACV SI Project'
    doc.core_properties.comments='Technical report for experiment results/final_run.'
    normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6)
    for level in ['Title','Heading 1','Heading 2']:
        doc.styles[level].font.color.rgb=RGBColor.from_string('17365D')
    section=doc.sections[0];section.top_margin=Inches(.75);section.bottom_margin=Inches(.75)
    title_seen=False;i=0;plain=[]
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        plain.append(clean_inline(line))
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=lines[i].strip()
                if not re.fullmatch(r'[|\s:\-]+',row):block.append([clean_inline(v.strip()) for v in row.strip('|').split('|')])
                i+=1
            table=doc.add_table(rows=1,cols=len(block[0]));table.style='Light Shading Accent 1'
            for cell,value in zip(table.rows[0].cells,block[0]):cell.text=value
            for row in block[1:]:
                for cell,value in zip(table.add_row().cells,row):cell.text=value
            plain.extend(' | '.join(row) for row in block)
            continue
        if line.startswith('# '):
            doc.add_heading(clean_inline(line[2:]),0 if not title_seen else 1);title_seen=True
        elif line.startswith('## '):doc.add_heading(clean_inline(line[3:]),1)
        elif line.startswith('### '):doc.add_heading(clean_inline(line[4:]),2)
        elif line.startswith('- '):doc.add_paragraph(clean_inline(line[2:]),style='List Bullet')
        else:doc.add_paragraph(clean_inline(line))
        i+=1
    doc.save(root/'docs/explanation_ru.docx')
    (root/'docs/explanation_ru.txt').write_text('\n\n'.join(plain)+'\n',encoding='utf-8')

if __name__=='__main__':main()
