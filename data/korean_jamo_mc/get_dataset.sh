#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$script_dir"

if [[ ! -s input.txt ]]; then
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
    print(f'OPUS-100 download via datasets failed ({exc}); writing a tiny fallback corpus.')
    out.write_text('English: Hello Korean: 안녕하세요.\nHello\t안녕하세요.\nGood morning\t좋은 아침입니다.\nKorean jamo lite\t한국어 조사 은는 예제입니다.\n', encoding='utf-8')
PY
fi

if ! grep -q "English: Hello Korean:" input.txt; then
  printf '\nEnglish: Hello Korean: 안녕하세요.\n' >> input.txt
fi

python3 ../template/utils/korean/extract_jamo_lite_streams.py input.txt . --metadata-json ''

lanes=(char first_jamo last_jamo eun_neun)
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
