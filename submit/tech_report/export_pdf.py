# -*- coding: utf-8 -*-
"""Open the DOCX in Word, update all fields (TOC + page numbers), repaginate,
then export to PDF. Word COM handles CJK fonts natively (no LibreOffice)."""
import os
import win32com.client as win32

DOCX = r'a:\VScode\Code\Projects\embodied_ai_contest\submit\tech_report\技术方案.docx'
PDF = r'a:\VScode\Code\Projects\embodied_ai_contest\submit\tech_report\技术方案.pdf'

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(DOCX)
    # update every field (including the TOC) so the contents populate
    doc.Fields.Update()
    for toc in doc.TablesOfContents:
        toc.Update()
        # \h hyperlink switch colors entries blue -> force black, no underline
        rng = toc.Range
        rng.Font.Color = 0          # wdColorBlack (BGR 0x000000)
        rng.Font.Underline = 0      # wdUnderlineNone
    doc.Repaginate()
    doc.Save()  # persist the populated TOC back into the docx
    doc.SaveAs(PDF, FileFormat=17)  # 17 = wdFormatPDF
    print('PDF exported:', PDF)
    doc.Close(False)
finally:
    word.Quit()
