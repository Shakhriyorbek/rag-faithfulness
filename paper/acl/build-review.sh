#!/usr/bin/env bash
# Builds main-review.pdf: the anonymous full-layout paper without nine
# appendix sections (the six always-appendix ones and the three robustness
# appendices between the REVIEWCOPY-OMIT markers in paper.tex), for AI
# review services that cap uploads at 35 pages. Not a submission build;
# build.sh is unchanged by this script.
#
# Step 1 compiles the ordinary anonymous build in a temp dir with
# intermediates kept, to get a fresh .aux. Step 2 copies the \newlabel lines
# of the omitted sections into review-labels.tex, so references to them print
# their full-paper letters instead of ??. Step 3 compiles with \REVIEWCOPY=1.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
OMIT="anchors esa qasper discussion-detail deploy choosing"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

cp -r paper.tex custom.bib acl.sty acl_natbib.bst sections "$tmp"/
printf '\\def\\COMPACTLAYOUT{0}\n\\def\\ANONYMOUS{1}\n\\input{paper}\n' > "$tmp/full.tex"
( cd "$tmp" && "$TECTONIC" -X compile full.tex -k >/dev/null )

labels=$( { for f in $OMIT; do grep -oh '\\label{[^}]*}' "sections/$f.tex"; done; \
            sed -n '/REVIEWCOPY-OMIT-BEGIN/,/REVIEWCOPY-OMIT-END/p' paper.tex \
              | grep -o '\\label{[^}]*}'; } \
         | sed 's/\\label{\(.*\)}/\1/' | sort -u)
: > review-labels.tex
for l in $labels; do
  grep -F "\\newlabel{$l}" "$tmp/full.aux" >> review-labels.tex \
    || { echo "label $l not found in full build" >&2; exit 1; }
done

printf '\\def\\COMPACTLAYOUT{0}\n\\def\\ANONYMOUS{1}\n\\def\\REVIEWCOPY{1}\n\\input{paper}\n' > main-review.tex
"$TECTONIC" -X compile main-review.tex --keep-logs >/dev/null
undef=$(grep -ciE 'undefined' main-review.log || true)
pages=$(pdfinfo main-review.pdf | awk '/Pages/{print $2}')
qq=$(pdftotext main-review.pdf - 2>/dev/null | grep -c '??' || true)
printf 'wrote main-review.pdf  %s pages  undefined-refs=%s  rendered-??=%s\n' "$pages" "$undef" "$qq"
rm -f main-review.tex review-labels.tex main-review.log main-review.blg
