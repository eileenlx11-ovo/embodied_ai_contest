# -*- coding: utf-8 -*-
"""Export the PPTX to PDF via PowerPoint COM for visual verification."""
import win32com.client as win32

PPTX = r'a:\VScode\Code\Projects\embodied_ai_contest\submit\tech_report\答辩PPT.pptx'
PDF = r'a:\VScode\Code\Projects\embodied_ai_contest\submit\tech_report\答辩PPT.pdf'

ppt = win32.gencache.EnsureDispatch('PowerPoint.Application')
try:
    deck = ppt.Presentations.Open(PPTX, WithWindow=False)
    deck.SaveAs(PDF, 32)  # 32 = ppSaveAsPDF
    deck.Close()
    print('PDF exported:', PDF)
finally:
    ppt.Quit()
