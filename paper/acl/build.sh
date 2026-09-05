#!/usr/bin/env bash
# Builds both variants of the ACL manuscript.
#   main.pdf           - "review": anonymous, line-numbered. This is the ARR
#                        submission file.
#   main-preprint.pdf  - "preprint": named, page-numbered. Reading copy.
#
# Needs tectonic (self-contained, no TeX Live):
#   curl -sSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz \
#     | tar -xz -C ~/.local/bin tectonic
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"

"$TECTONIC" -X compile main.tex --keep-logs
echo "wrote main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages, anonymous)"

sed 's/\\usepackage\[review\]{acl}/\\usepackage[preprint]{acl}/' main.tex > main-preprint.tex
"$TECTONIC" -X compile main-preprint.tex
rm -f main-preprint.tex
echo "wrote main-preprint.pdf ($(pdfinfo main-preprint.pdf | awk '/Pages/{print $2}') pages, named)"
