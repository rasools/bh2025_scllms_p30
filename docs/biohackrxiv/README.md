BioHackrXiv report files live here. build locally with pandoc.

basic build

```
cd docs/biohackrxiv
pandoc report.md \
  --from markdown+yaml_metadata_block+citations \
  --metadata-file metadata.yaml \
  --citeproc \
  --bibliography references.bib \
  -o biohackrxiv_report.pdf
```

install pandoc

```
brew install pandoc
```

optional latex if pdf complains

```
brew install --cask mactex-no-gui
```

submit the pdf on osf preprints under BioHackrXiv. pick a license, add authors, link repo url and a zenodo doi if you have one.


