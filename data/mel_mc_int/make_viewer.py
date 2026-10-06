#!/usr/bin/env python3
"""Static, safely quoted report for a completed mel continuation run."""
import argparse
import html
import json
from pathlib import Path

# Kept as literal JavaScript, outside Python interpolation/escape processing.
COMMAND_JS = r'''
function shellQuote(value) {
  return "'" + String(value).replaceAll("'", "'\\''") + "'";
}
function commandFor(s) {
  const cutoff = Number(s.cutoff_s), frames = Number(s.max_new_tokens);
  const temperature = Number(s.temperature), topk = Number(s.top_k), seed = Number(s.seed);
  if (!Number.isFinite(cutoff) || cutoff <= 0 || !Number.isInteger(frames) || frames < 1 ||
      !Number.isFinite(temperature) || temperature <= 0 || !Number.isInteger(topk) || topk < 1 ||
      !Number.isInteger(seed) || !s.input_audio.trim()) throw new Error('Check the path and numeric fields.');
  return ['env', 'MEL_MC_DEVICE=' + s.device, 'MEL_MC_DTYPE=' + s.dtype,
          'MEL_MC_TEMPERATURE=' + temperature, 'MEL_MC_TOP_K=' + topk, 'MEL_MC_SEED=' + seed,
          'bash', 'data/mel_mc_int/demo_infer.sh', s.out_dir, s.input_audio,
          cutoff, frames, '--manifest', s.manifest].map(shellQuote).join(' ');
}
'''


def build_viewer(output_dir, settings):
    output_dir = Path(output_dir)
    prompt = settings.get('prompt', {})
    boundary = prompt.get('generation_start_s', 0.)
    fields = [('input_audio', 'Audio filesystem path'), ('out_dir', 'Checkpoint directory'),
              ('manifest', 'Dataset manifest'), ('cutoff_s', 'Cutoff seconds'),
              ('max_new_tokens', 'New mel frames'), ('device', 'Device'), ('dtype', 'Dtype'),
              ('temperature', 'Temperature'), ('top_k', 'Top-k (1 is greedy)'), ('seed', 'Seed')]
    inputs = ''.join(f'<label>{label}<input id="{key}" value="{html.escape(str(settings.get(key, "")), quote=True)}"></label>'
                     for key, label in fields)
    data = json.dumps(settings, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    document = '''<!doctype html><meta charset="utf-8"><title>Mel continuation</title>
<style>body{font:16px system-ui;max-width:960px;margin:2rem;line-height:1.5}audio,input{width:100%}
section{border:1px solid #ccc;padding:1rem;margin:1rem 0}label{display:block;margin:.5rem 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:1rem}</style>
<h1>Mel audio continuation</h1>
<section><h2>Original decoded audio prefix</h2><audio controls src="original_prefix.wav"></audio></section>
<section><h2>Codec-only prompt reconstruction</h2><audio controls src="codec_prompt.wav"></audio></section>
<section><h2>Reconstructed prompt plus generated continuation</h2><audio controls src="generated.wav"></audio>
<p>Generated mel frames begin at ''' + html.escape(f'{boundary:.3f}') + ''' seconds. Overlapping inverse windows can blend audio around that frame boundary.</p></section>
<section><h2>Continuation-only audition</h2><audio controls src="continuation.wav"></audio></section>
<p>Conditioned frames: ''' + html.escape(str(settings.get('conditioned_frames', 'unknown'))) + '''. Model context: ''' + html.escape(str(settings.get('context_seconds', 'unknown'))) + ''' seconds.</p>
<p><a href="generated.csv">Generated states</a> · <a href="generated.mel.csv">Mel container</a> · <a href="run.json">Run settings</a></p>
<section><h2>Next run</h2><p>Paste the generated command into a shell at the repository root.
A browser picker cannot expose your filesystem path; enter that path explicitly below.</p>
<label>Audition a local file <input id="file" type="file" accept="audio/*"></label><audio id="preview" controls></audio>''' + inputs + '''<pre id="command"></pre></section>
<script>const initial = ''' + data + ';\n' + COMMAND_JS + r'''
const keys = ['input_audio','out_dir','manifest','cutoff_s','max_new_tokens','device','dtype','temperature','top_k','seed'];
const command = document.getElementById('command');
function update() {
  const values = Object.fromEntries(keys.map(key => [key, document.getElementById(key).value]));
  try { command.textContent = commandFor(values); }
  catch (error) { command.textContent = error.message; }
}
let objectURL;
document.getElementById('file').addEventListener('change', event => {
  const file = event.target.files[0]; if (!file) return;
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = URL.createObjectURL(file); document.getElementById('preview').src = objectURL;
});
for (const key of keys) document.getElementById(key).addEventListener('input', update);
update();
</script>'''
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'index.html').write_text(document, encoding='utf-8')
    return output_dir/'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--run_json', help='Defaults to OUTPUT_DIR/run.json')
    args = parser.parse_args()
    settings = json.loads(Path(args.run_json or Path(args.output_dir)/'run.json').read_text())
    print(build_viewer(args.output_dir, settings))


if __name__ == '__main__':
    main()
