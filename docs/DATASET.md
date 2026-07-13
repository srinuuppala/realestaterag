# Real Estate Knowledge Base Dataset

92 documents across 4 file formats, covering all required categories for
the Real Estate AI Assistant RAG knowledge base.

## Format breakdown
| Format | Count | Folder |
|---|---|---|
| PDF | 21 | `pdf/` |
| DOCX | 21 | `docx/` |
| HTML | 23 | `html/` |
| Markdown | 27 | `markdown/` |
| **Total** | **92** | |

## Category coverage

| Required category | Where it lives |
|---|---|
| Project brochures | `pdf/*_brochure.pdf` (6 projects) |
| Builder websites | `html/*_home.html`, `*_about.html` (3 builders) |
| Property listing portals | `html/propertybazaar_listing_*.html` (6 listings) |
| Builder profile documents | `pdf/*_builder_profile.pdf`, `html/*_about.html` |
| RERA documentation | `pdf/*_rera_summary.pdf` (6), `markdown/rera_general_information.md` |
| Privacy Policy | `html/*_privacy_policy.html` (4, incl. portal) |
| Terms & Conditions | `html/*_terms_conditions.html` (4, incl. portal) |
| FAQ pages | `html/*_faq.html` (3 builders) |
| Payment plans | `pdf/*_payment_plan.pdf` (6 projects) |
| Cancellation & Refund policies | `docx/*_cancellation_refund_policy.docx` (3 builders) |
| Home loan information | `markdown/home_loan_*.md` (5 docs) |
| Registration process | `docx/*_registration_process.docx` (6 projects) |
| Possession guidelines | `docx/*_possession_guidelines.docx` (6 projects) |
| Customer support documentation | `markdown/*_customer_support.md` (3 builders) |
| *(bonus)* Amenities guides | `markdown/*_amenities_guide.md` (6 projects) |
| *(bonus)* Location guides | `markdown/*_location_guide.md` (6 projects) |
| *(bonus)* Floor plan descriptions | `markdown/*_floor_plans.md` (6 projects) |
| *(bonus)* Sale agreement / legal terms | `docx/*_sale_agreement_terms.docx` (6 projects) |

## Content notes

- Content is built around 3 fictional builders (Skyline Horizon Developers,
  Meridian Greens Realty, Urban Nest Infrastructures) and 6 fictional
  projects, kept internally consistent (prices, RERA numbers, possession
  dates, locations) across every document type so retrieval/citation
  behavior looks realistic — cross-document questions (e.g. "what's the
  payment plan for the project priced under 1 crore in Hinjewadi?") have a
  real, traceable answer.
- All RERA numbers, prices, and dates are synthetic and for demo purposes
  only — do not use as real regulatory or financial data.
- Generated with `generate.py` (included) — rerun it any time to regenerate
  or extend the set (e.g. add more builders/projects to the `BUILDERS` dict
  at the top of the script).

## Using with the RAG pipeline

Drop the four folders' contents into `data/raw/` of the RAG assistant repo,
then run:
```
python -m src.ingestion.build_index --reset
```
