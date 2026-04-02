# SPIC LaTeX Paper Build Instructions

## Overview

This directory contains a preview manuscript for submission to **Signal Processing: Image Communication (SPIC)** following the Elsevier `elsarticle` template.

## Directory Structure

```
paper_spic/
├── main.tex              # English main document
├── main.pdf              # English compiled PDF
├── main_cn.tex           # Chinese main document
├── main_cn.pdf           # Chinese compiled PDF
├── sections/             # English paper sections
├── sections_cn/          # Chinese paper sections
├── figures/              # Figure assets used by both papers
├── tables/               # Optional standalone tables
├── refs.bib              # Bibliography
├── scripts/              # Figure generation scripts
├── data/phase1/          # Phase 1 supplementary results and tables
├── archive/legacy_chinese/ # Older Chinese draft kept for reference
├── build_aux/            # Auxiliary build files (.aux/.log/.bbl, etc.)
├── versions/             # Versioned snapshots with PDFs and notes
├── notes/                # Changelog and polishing notes
└── README_BUILD.md       # This file
```

## Compilation Instructions

### Prerequisites

Ensure you have a LaTeX distribution installed:
- **TeX Live** (Linux/Mac): `sudo apt-get install texlive-full` or `brew install --cask mactex`
- **MiKTeX** (Windows): https://miktex.org/

### Build Steps

```bash
# 1. First pdflatex run
cd paper_spic
pdflatex main.tex

# 2. Generate bibliography
bibtex main

# 3. Second pdflatex run (resolves references)
pdflatex main.tex

# 4. Third pdflatex run (final pass)
pdflatex main.tex
```

### One-Command Build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Using Latexmk (Recommended)

```bash
latexmk -pdf main.tex
```

### Chinese Build

```bash
cd paper_spic
xelatex main_cn.tex
xelatex main_cn.tex
```

To clean auxiliary files:
```bash
latexmk -c
```

Note: this repository now stores existing auxiliary outputs under `build_aux/`, but a fresh local compilation may still regenerate temporary files in the root directory depending on the build command you use.

## Current Status

### What Works

✅ **Document structure**: Complete with all 7 sections
✅ **Tables**: All tables (1-3) are fully coded in LaTeX
✅ **References**: BibTeX file with ~15 verified references
✅ ** elsarticle format**: Preprint mode with correct frontmatter

### Figures (✅ Completed)

All figures have been generated and integrated:

| Figure | Description | File | Status |
|--------|-------------|------|--------|
| Figure 1 | Mechanism-driven view of KD under visibility degradation | `figures/figure1_conceptual.pdf` | ✅ |
| Figure 2 | Branch-wise performance across visibility levels | `figures/figure2_branch_performance.pdf` | ✅ |
| Figure 3 | Gain relative to student-only baseline | `figures/figure3_gains.pdf` | ✅ |
| Figure 4 | Mechanism analysis (three-panel) | `figures/figure4_mechanism_analysis.pdf` | ✅ |

**Figure regeneration**: If needed, run:
```bash
cd paper_spic
python scripts/generate_figures.py
```

### TODO Items in Manuscript

🔲 **Author information**: Replace red TODO markers in `main.tex`
   - Author names
   - Email addresses
   - Affiliations
   - Acknowledgments

🔲 **Additional references**: See TODO comments in:
   - `sections/related_work.tex` (lines 42-47)
   - `refs.bib` (bottom TODO list)

## Known Issues

1. **Compilation warnings**: Expected on first build due to undefined references; resolved after bibtex + 2x pdflatex
2. **Figure placement**: Currently uses `\fbox` placeholders; replace with `\includegraphics` when figures are ready
3. **Hyperref colors**: URLs may appear colored; adjust `hyperref` options in `main.tex` if needed

## Submission Checklist

Before submission, ensure:

- [ ] Replace all red TODO markers with actual content
- [ ] Generate final figures (PDF or PNG format)
- [ ] Update figure paths in tex files
- [ ] Complete reference list (verify all citations resolve)
- [ ] Check compilation produces no errors
- [ ] Verify PDF meets SPIC formatting requirements
- [ ] Add ORCID IDs if available

## Paper Narrative Summary

**Core Claim**: KD failure under visibility degradation is dominated by visibility disruption (occlusion), while commonly assumed explanations (distribution mismatch, uncertainty amplification) fail to account for observed behavior.

**Key Evidence**:
- Branch-wise structure exists (Table 1, 2)
- Distribution mismatch: r=+0.20, p=0.87 (insufficient)
- Uncertainty amplification: r=+0.16-0.41, p>0.7 (insufficient)
- Visibility disruption (occlusion): r=-0.989, p=0.0015 (supported)

**Target Audience**: SPIC readers interested in robust object detection, knowledge distillation, and adverse weather conditions.

## Contact

For questions about this manuscript, contact the corresponding author (TODO: add email).

---

**Last Updated**: 2026-04-02
**Status**: Preview Manuscript (Pre-submission)
