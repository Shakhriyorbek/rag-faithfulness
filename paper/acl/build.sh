#!/usr/bin/env bash
# Builds all four variants from ONE source: paper.tex + sections/.
#
#   main.pdf                   full layout,    anonymous + line-numbered (ARR)
#   main-preprint.pdf          full layout,    named           (reading copy)
#   main-8page.pdf             compact layout, anonymous + line-numbered
#   main-8page-preprint.pdf    compact layout, named
#
# "Compact" moves the thirteen secondary sections into the appendix, which ACL
# does not count toward the page limit. Both layouts contain every sentence;
# only placement differs, which is the point -- two hand-maintained .tex files
# drifted twice without a build error.
#
# Needs tectonic (self-contained, no TeX Live):
#   curl -sSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-gnu.tar.gz \
#     | tar -xz -C ~/.local/bin tectonic
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"

build () {  # name, compact, anonymous
  local name="$1" compact="$2" anon="$3"
  printf '\\def\\COMPACTLAYOUT{%s}\n\\def\\ANONYMOUS{%s}\n\\input{paper}\n' \
         "$compact" "$anon" > "$name.tex"
  "$TECTONIC" -X compile "$name.tex" --keep-logs >/dev/null
  local undef pages qq
  undef=$(grep -ciE 'undefined' "$name.log" || true)
  pages=$(pdfinfo "$name.pdf" | awk '/Pages/{print $2}')
  qq=$(pdftotext "$name.pdf" - 2>/dev/null | grep -c '??' || true)
  printf 'wrote %-24s %2s pages  undefined-refs=%s  rendered-??=%s\n' \
         "$name.pdf" "$pages" "$undef" "$qq"
  # Undefined citations and refs only ever appear in the LOG, never on
  # tectonic's stdout. Checking stdout reported "0 undefined" for a file that
  # had four dangling citations and four ?? in the PDF.
  if [ "$undef" != "0" ] || [ "$qq" != "0" ]; then
    echo "  ^ FAILED: see $name.log" >&2
    grep -iE 'undefined' "$name.log" | head -5 >&2 || true
    exit 1
  fi
}

build main            0 1
build main-preprint   0 0
build main-8page      1 1
build main-8page-preprint 1 0

# main.tex and main-8page.tex are committed wrappers; the others are generated.
rm -f main-preprint.tex main-8page-preprint.tex
