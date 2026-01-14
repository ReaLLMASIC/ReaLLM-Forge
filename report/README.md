# Reports

This folder is for formalizing reports from results from the repo.

Many of the analysis scripts facilitate the creation of images, and can be
easily ported into a new report folder.

Keep in mind these are only templates, and many conferences have rules against
including personally identifiable information into repos included during peer
review.

## Setting Up Tectonic

The main dependency for this folder is Tectonic.

Tectonic is a rust based LaTeX formatter, creating both the pdf, intermediate
files, and also detecting and installing package dependencies.

Use the following will obtain tectonic on a Unix sytem:
```sh
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net |sh
```

One can also utilize normal `cargo install tectonic` if you have cargo
installed.

## Using the framework

These folders utilize tectonic for LaTeX report writing, and have several
scripts for automating the report creation process.

- `export_to_arxiv.sh` -- exports directly into arxiv zip file format
- `build.sh` -- initializes build from tex to pdf
- `watch.sh` -- if run, saved buffers will initialie build.sh on the pdf for instant feedback
- `clean.sh` -- removing auxialiary files and zip files
- `rename.sh` -- automatic renaming of filenames and references within the folder
