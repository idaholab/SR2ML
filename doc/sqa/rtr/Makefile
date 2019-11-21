LATEX_FLAGS=-interaction=nonstopmode

all: default sr2ml_requirements_traceability_matrix.pdf

sr2ml_requirements_traceability_matrix.pdf : sr2ml_requirements_traceability_matrix.tex sr2ml_requirements_traceability_matrix.bib intro.tex ../../version.tex ../sdd/sr2mlDepAndLimitations.tex
	pdflatex $(LATEX_FLAGS) sr2ml_requirements_traceability_matrix.tex
	pdflatex $(LATEX_FLAGS) sr2ml_requirements_traceability_matrix.tex
	pdflatex $(LATEX_FLAGS) sr2ml_requirements_traceability_matrix.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex
	python createSQAtracebilityMatrix.py -i ../srs/requirements_list.xml -o traceability_matrix.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof sr2ml_requirements_traceability_matrix.pdf
