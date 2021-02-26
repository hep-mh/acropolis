mkdir -p plots
cp ../plots/*.pdf plots
tar cvzf src.tar.gz JHEP.bst jheppub.sty manual.tex references.bib plots/
rm -rf plots/
