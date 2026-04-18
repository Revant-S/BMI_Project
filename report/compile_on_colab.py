# ══════════════════════════════════════════════════════════════
# Compile the LaTeX Report to PDF on Google Colab
#
# Instructions:
#   1. Run this script in a Colab cell after cloning the repo.
#   2. The PDF will be downloaded to your computer automatically.
# ══════════════════════════════════════════════════════════════

# Install TeX Live (minimal, fast install ~60 seconds)
!apt-get install -y texlive-latex-extra texlive-science 2>&1 | tail -5

import os
os.chdir('/content/BMI_Project/report')

# Compile twice (needed for table of contents page numbers)
!pdflatex -interaction=nonstopmode ecg_arrhythmia_report.tex
!pdflatex -interaction=nonstopmode ecg_arrhythmia_report.tex

# Download the PDF
from google.colab import files
files.download('ecg_arrhythmia_report.pdf')
print("PDF downloaded!")
