#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$script_dir"

mkdir -p raw
if [[ ! -s input.txt || $(wc -c < input.txt) -lt 500 ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
from pathlib import Path
out = Path('input.txt')
try:
    from datasets import load_dataset
    ds = load_dataset('opus100', 'en-ko', split='train', streaming=True)
    with out.open('w', encoding='utf-8') as f:
        for i, row in enumerate(ds):
            ko = row.get('translation', {}).get('ko', '')
            en = row.get('translation', {}).get('en', '')
            if ko:
                f.write((en + '\t' if en else '') + ko.replace('\n', ' ') + '\n')
            if i >= 200000:
                break
except Exception as exc:
    print(f'OPUS-100 download via datasets failed ({exc}); writing a fallback corpus.')
    out.write_text(('Hello\t안녕하세요.\nGood morning\t좋은 아침입니다.\nKorean multicontext\t한국어 다중 문맥 예제입니다.\n' * 50), encoding='utf-8')
PY
  fi
fi

python3 ../template/utils/korean/extract_multicontext_streams.py input.txt . --use-pos --metadata-json '' --metadata-yaml ''

lanes=(script choseong jungseong jongseong jung_base1 jung_base2 jung_has_w jung_has_y jung_has_i jong_base1 jong_base2 jong_base3 choseong_tense choseong_aspirated choseong_nasal_liquid choseong_place jung_height jung_backness jung_round jong_complex has_batchim syllable_index_mod codepoint_mod pos char)
for lane in "${lanes[@]}"; do
  (
    cd "$lane"
    python3 ../../template/prepare.py -t input.txt --method char -s -S "$lane"
    prepared_dir="char_${lane}"
    cp "${prepared_dir}/meta.pkl" meta.pkl
    cp "${prepared_dir}/train.bin" train.bin
    cp "${prepared_dir}/val.bin" val.bin
  )
done
