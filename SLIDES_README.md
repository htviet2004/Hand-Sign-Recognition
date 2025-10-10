How to use the slides

- The `slides.md` file is a lightweight Markdown slide deck you can edit.
- To export to PDF or PPTX, install Pandoc + LaTeX (for PDF) or use Marp/Reveal if you prefer.

Quick export commands (PowerShell):

# Using Pandoc -> PDF (requires LaTeX):
# pandoc slides.md -t beamer -o slides.pdf

# Using Pandoc -> PowerPoint
# pandoc slides.md -t pptx -o slides.pptx

Speaker notes: edit the notes in `slides.md` under each slide.
