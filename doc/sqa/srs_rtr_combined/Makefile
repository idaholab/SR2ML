LATEX_FLAGS=-interaction=nonstopmode

all: sr2ml_software_requirements_specifications_and_traceability.pdf

sr2ml_software_requirements_specifications_and_traceability.pdf : sr2ml_software_requirements_specifications_and_traceability.tex sr2ml_software_requirements_specifications_and_traceability.bib intro.tex ../../version.tex ../sdd/sr2mlDepAndLimitations.tex
	python ../srs/readRequirementsAndCreateLatex.py -i ../srs/requirements_list.xml -o ../srs/requirements.tex
	python ../rtr/createSQAtracebilityMatrix.py -i ../srs/requirements_list.xml -o ../rtr/traceability_matrix.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_requirements_specifications_and_traceability.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_requirements_specifications_and_traceability.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_requirements_specifications_and_traceability.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex
	python ../../../scripts/TestHarness/testers/RavenUtils.py --manual-list > dependencies.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof sr2ml_software_requirements_specifications_and_traceability.pdf
