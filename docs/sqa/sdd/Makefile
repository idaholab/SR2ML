LATEX_FLAGS=-interaction=nonstopmode

all: default sr2ml_software_design_description.pdf

sr2ml_software_design_description.pdf : sr2ml_software_design_description.tex sr2ml_software_design_description.bib sr2mlDesignStakeHoldersAndConcerns.tex sr2mlIntro.tex ../../version.tex sr2mlSoftwareDesign.tex sr2mlDepAndLimitations.tex otherDesignDocs.tex dataInterfaceSystemControl.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_design_description.tex
	bibtex sr2ml_software_design_description
	pdflatex $(LATEX_FLAGS) sr2ml_software_design_description.tex
	pdflatex $(LATEX_FLAGS) sr2ml_software_design_description.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof sr2ml_software_design_description.pdf dependencies.tex
