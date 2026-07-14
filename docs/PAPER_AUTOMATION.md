# Paper Automation

Manual content lives in `latex/main.tex`, `latex/supplementary.tex`, `latex/reference.bib`, `latex/sections`, `latex/supplement`, `latex/tables/manual`, and `latex/figures/manual`. Repeated exports may write only `latex/tables/generated`, `latex/figures/generated`, and `latex/generated`.

Run aggregation, dashboard generation, paper export, draft build, and final checks in that order. Drafts render explicit missing statuses. Final checks reject result TODOs, TBDs, NaNs, blockers, missing required artifacts, and failed strict audits. See [PAPER_WORKFLOW.md](PAPER_WORKFLOW.md).
