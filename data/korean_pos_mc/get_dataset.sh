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
    sentences = set()
    ds_opus = load_dataset('opus100', 'en-ko', split='train')
    for i, row in enumerate(ds_opus):
        ko = row.get('translation', {}).get('ko', '')
        en = row.get('translation', {}).get('en', '')
        if ko:
            sentences.add((en + '\t' if en else '') + ko.replace('\n', ' '))
    for cfg in ['dp', 'ner', 'mrc', 'nli', 're', 'sts', 'ynat']:
        try:
            ds_klue = load_dataset('klue', cfg)
            for split in ds_klue.keys():
                for row in ds_klue[split]:
                    for k in ['sentence', 'sentence_form', 'title', 'article', 'premise', 'hypothesis', 'text']:
                        val = row.get(k)
                        if isinstance(val, str) and len(val.strip()) > 0:
                            sentences.add(val.strip().replace('\n', ' '))
                    if 'word_form' in row and isinstance(row['word_form'], list):
                        s = ' '.join(row['word_form']).strip()
                        if s:
                            sentences.add(s)
        except Exception as e:
            print(f'Error loading klue/{cfg}: {e}')
    with out.open('w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
except Exception as exc:
    print(f'Dataset download via datasets failed ({exc}); writing a fallback corpus.')
    out.write_text(('Hello\t안녕하세요.\nGood morning\t좋은 아침입니다.\nKorean multicontext\t한국어 다중 문맥 예제입니다.\n' * 50), encoding='utf-8')
PY
  fi
fi

if [[ ! -f char/train.bin || ! -f pos/train.bin ]]; then
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
else
  echo "Dataset streams already prepared in $script_dir. Skipping extraction."
fi

