.PHONY: main supplementary draft check-final final clean

main:
	latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error -outdir=build main.tex

supplementary:
	latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error -outdir=build supplementary.tex

draft: main supplementary
	python ../scripts/run.py research paper-check --paper-root . --results-root ../result --mode draft

check-final:
	python ../scripts/run.py research paper-check --paper-root . --results-root ../result --mode final

final: check-final main supplementary

check:
	python ../scripts/run.py research paper-check --paper-root . --results-root ../result --mode draft

clean:
	latexmk -C -outdir=build main.tex
	latexmk -C -outdir=build supplementary.tex
