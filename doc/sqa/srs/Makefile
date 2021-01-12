LATEX_FLAGS=-interaction=nonstopmode

all: sr2ml_software_requirements_specifications.pdf

sr2ml_software_requirements_specifications.pdf : sr2ml_software_requirements_specifications.tex sr2ml_software_requirements_specifications.bib intro.tex ../../version.tex ../sdd/sr2mlDepAndLimitations.tex
	python readRequirementsAndCreateLatex.py -i requirements_list.xml -o requirements.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_requirements_specifications.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_requirements_specifications.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_requirements_specifications.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof sr2ml_software_requirements_specifications.pdf
