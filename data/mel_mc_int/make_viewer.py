#!/usr/bin/env python3
"""Write a small static HTML report for mel_mc_int continuation runs."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output_dir', required=True)
p.add_argument('--input_audio', required=True)
p.add_argument('--cutoff_s', required=True)
args = p.parse_args()
out = Path(args.output_dir)
body = f'''<!doctype html>
<meta charset="utf-8">
<title>Mel MC continuation viewer</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem;max-width:960px;line-height:1.45}}
audio{{width:100%}} .card{{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0}}
code,pre{{background:#f6f8fa;border-radius:6px;padding:.15rem .35rem}} pre{{padding:1rem;overflow:auto}}
label{{display:block;margin:.5rem 0}}
</style>
<h1>Mel multicontext audio continuation</h1>
<p>Source: <code>{html.escape(str(args.input_audio))}</code>; cutoff: <strong>{html.escape(str(args.cutoff_s))}s</strong>.</p>
<div class="card"><h2>Reconstructed inference / continuation</h2><audio controls src="generated.wav"></audio></div>
<div class="card"><h2>Generated artifacts</h2><p><a href="generated.csv">generated.csv</a> · <a href="generated.mel.csv">self-describing mel CSV</a></p></div>
<div class="card">
  <h2>Select the next input audio and cutoff</h2>
  <p>This viewer is static, so it cannot run Python by itself. Use it to audition a local file, choose the cutoff, then copy the command below into the repo shell.</p>
  <label>Audio file <input id="file" type="file" accept="audio/*"></label>
  <audio id="preview" controls></audio>
  <label>Cutoff seconds <input id="cutoff" type="number" min="0" step="0.01" value="{html.escape(str(args.cutoff_s))}"></label>
  <label>Run output directory <input id="outdir" value="out/mel_mc_int"></label>
  <label>Max new mel frames <input id="frames" type="number" min="1" step="1" value="200"></label>
  <pre id="cmd">bash data/mel_mc_int/demo_infer.sh out/mel_mc_int path/to/audio.wav {html.escape(str(args.cutoff_s))} 200</pre>
</div>
<script>
const file = document.getElementById('file'), preview = document.getElementById('preview');
const cutoff = document.getElementById('cutoff'), outdir = document.getElementById('outdir'), frames = document.getElementById('frames'), cmd = document.getElementById('cmd');
let name = 'path/to/audio.wav';
function q(s) {{ return '"' + String(s).replaceAll('"', '\\"') + '"'; }}
function update() {{ cmd.textContent = `bash data/mel_mc_int/demo_infer.sh ${{q(outdir.value)}} ${{q(name)}} ${{q(cutoff.value)}} ${{q(frames.value)}}`; }}
file.addEventListener('change', () => {{
  const f = file.files && file.files[0];
  if (!f) return;
  name = f.name;
  preview.src = URL.createObjectURL(f);
  update();
}});
for (const el of [cutoff, outdir, frames]) el.addEventListener('input', update);
update();
</script>
'''
(out / 'index.html').write_text(body, encoding='utf-8')
print(out / 'index.html')
