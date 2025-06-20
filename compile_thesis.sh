#!/bin/bash

# Script to compile the thesis.tex LaTeX file
# This script assumes you have a working LaTeX distribution (e.g., TeX Live)
# and BibTeX installed.

# Define the main LaTeX filename (without .tex extension)
FILENAME="thesis"

echo "Starting LaTeX compilation for ${FILENAME}.tex..."

# Initial pdflatex run: generates .aux file needed by bibtex, .toc, .lof, .lot
echo "Running pdflatex (1st pass)..."
pdflatex -interaction=nonstopmode "${FILENAME}.tex"
if [ $? -ne 0 ]; then
    echo "Error during first pdflatex run. Please check ${FILENAME}.log for details."
    # Do not exit here, allow user to see subsequent errors or partial success
    # exit 1
fi

# BibTeX run: processes .aux file and .bib file to create .bbl file
echo "Running bibtex..."
bibtex "${FILENAME}.aux"
if [ $? -ne 0 ]; then
    echo "Warning: Error or issue during bibtex run. Please check ${FILENAME}.blg for details."
    # Common issue: .bib file not found or syntax errors in .bib, or no citations yet.
    # This is not a fatal error for PDF generation if no citations are strictly needed yet,
    # but the bibliography will be missing/incorrect.
    echo "Continuing pdflatex runs, but bibliography may be affected."
fi

# Second pdflatex run: incorporates bibliography (.bbl) and updates ToC, LoF, LoT
echo "Running pdflatex (2nd pass)..."
pdflatex -interaction=nonstopmode "${FILENAME}.tex"
if [ $? -ne 0 ]; then
    echo "Error during second pdflatex run. Please check ${FILENAME}.log for details."
    # Do not exit here
fi

# Third pdflatex run: ensures all cross-references (labels, citations, ToC page numbers) are correct
echo "Running pdflatex (3rd pass)..."
pdflatex -interaction=nonstopmode "${FILENAME}.tex"
if [ $? -ne 0 ]; then
    echo "Error during third pdflatex run. Please check ${FILENAME}.log for details."
    # Do not exit here, final PDF might still be generated but with errors
fi

echo "Compilation process completed."
echo "Output PDF should be ${FILENAME}.pdf"
echo "Log file is ${FILENAME}.log"

if [ -f "${FILENAME}.pdf" ]; then
    echo "File ${FILENAME}.pdf created successfully."
else
    echo "Warning: ${FILENAME}.pdf was not created. Check logs for critical errors."
fi

echo "Please check ${FILENAME}.log (and ${FILENAME}.blg if bibtex issues) for any warnings or errors."

# Optional: Clean up auxiliary files
# echo "Cleaning up auxiliary files..."
# rm -f ${FILENAME}.aux ${FILENAME}.bbl ${FILENAME}.blg ${FILENAME}.lof ${FILENAME}.log ${FILENAME}.lot ${FILENAME}.out ${FILENAME}.toc
# Be careful with automatic cleanup, user might want to inspect these files.
# For now, it's commented out. The user can add it if they wish.

# Script finishes without explicit exit 0, implies success if commands before didn't cause termination
# due to `set -e` (which is not set here by default).
# A non-zero exit code from pdflatex will be propagated if `set -e` was active
# or if the user checks $? after running this script.
# For robustness in a pipeline, one might check $? after each pdflatex and exit.
# But for interactive use, allowing all passes to run can be helpful.
