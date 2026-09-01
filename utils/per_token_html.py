"""Generate small, independent HTML pages for per-token Plotly graphs."""

import html
import json
import math
import os

PLOTLY = "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"


def _json(value):
    return json.dumps(value, allow_nan=True).replace("</", "<\\/")


def _shell(title, payload, controls, script):
    return f"""<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>
{PLOTLY}<h1>{html.escape(title)}</h1><p><a href='per_token_metrics.html'>Report index</a></p>
{controls}<div id='error' style='color:#b00020;font-weight:bold'></div><div id='plot' style='height:82vh'></div>
<script>const rows={_json(payload)};
function plot(traces,layout){{if(typeof Plotly==='undefined'){{error.textContent='Plotly failed to load; check access to cdn.plot.ly.';return;}}Plotly.newPlot('plot',traces,layout).catch(e=>{{error.textContent=String(e);console.error(e);}});}}
{script}</script>"""


def _latest(rows):
    latest_iteration = {}
    for row in rows:
        latest_iteration[row["dataset"]] = max(
            latest_iteration.get(row["dataset"], -math.inf), row["iteration"]
        )
    return [row for row in rows if row["iteration"] == latest_iteration[row["dataset"]]]


def _write(path, content):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_static_slideshow(output_dir, rows):
    """Write keyboard/button navigation across validation PNG snapshots."""
    slides = []
    for dataset in sorted({row["dataset"] for row in rows}):
        safe_dataset = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
        for iteration in sorted({row["iteration"] for row in rows if row["dataset"] == dataset}):
            slides.append({
                "dataset": dataset,
                "iteration": iteration,
                "images": [
                    f"per_token_static_{safe_dataset}_iter_{iteration:08d}_by_{label}.png"
                    for label in ("frequency", "validation_loss", "training_loss",
                                  "vector_magnitude", "minimum_pairwise_angle")
                ],
            })
    html_page = f"""<!doctype html><meta charset='utf-8'><title>Per-token static snapshots</title>
<h1>Per-token static snapshots</h1><p><a href='per_token_metrics.html'>Report index</a></p>
<button id='previous'>← Previous</button> <button id='next'>Next →</button> <strong id='position'></strong>
<div id='images'></div><script>const slides={_json(slides)};let index=0;
function draw(){{const slide=slides[index];position.textContent=`${{slide.dataset}} — iteration ${{slide.iteration}} (${{index+1}}/${{slides.length}})`;images.replaceChildren();slide.images.forEach(path=>{{const image=document.createElement('img');image.src=path;image.style.cssText='display:block;max-width:100%;margin:1rem auto';images.appendChild(image);}});}}
function move(delta){{index=(index+delta+slides.length)%slides.length;draw();}}previous.onclick=()=>move(-1);next.onclick=()=>move(1);document.onkeydown=event=>{{if(event.key==='ArrowLeft')move(-1);if(event.key==='ArrowRight')move(1);}};draw();</script>"""
    _write(os.path.join(output_dir, "per_token_static_slideshow.html"), html_page)


def _overview(output_dir, filename, title, rows, metric, descending, dual_axis):
    payload = [[r["dataset"], r["token_id"], r["token_text_escaped"], r[metric],
                r["training_seen_count"], r["val_loss"], r["train_loss"]] for r in rows]
    right = "<label><input id='right' type='checkbox'> right logarithmic</label>" if dual_axis else ""
    if metric == "training_seen_count":
        secondary = ("traces.push({x:labels,y:d.map(r=>r[5]),name:'validation loss',mode:'markers',yaxis:'y2'},"
                     "{x:labels,y:d.map(r=>r[6]),name:'sampled training loss',mode:'markers',yaxis:'y2'});")
        right_title = "loss"
    else:
        secondary = ("traces.push({x:labels,y:d.map(r=>r[4]),name:'training occurrences',"
                     "mode:'markers',yaxis:'y2'});") if dual_axis else ""
        right_title = "training occurrences"
    y2 = (f"yaxis2:{{title:{_json(right_title)},type:right.checked?'log':'linear',"
          "overlaying:'y',side:'right'},") if dual_axis else ""
    script = f"""const ds=document.getElementById('dataset'),left=document.getElementById('left'){",right=document.getElementById('right')" if dual_axis else ""};new Set(rows.map(r=>r[0])).forEach(x=>ds.add(new Option(x,x)));
function draw(){{const d=rows.filter(r=>r[0]===ds.value&&Number.isFinite(r[3])).sort((a,b)=>{'b[3]-a[3]' if descending else 'a[3]-b[3]'}),labels=d.map(r=>`token ${{r[1]}} '${{r[2]}}'`),traces=[{{x:labels,y:d.map(r=>r[3]),name:{_json(title)},mode:'markers'}}];{secondary}
plot(traces,{{title:{_json(title)},xaxis:{{title:'token'}},yaxis:{{title:{_json(title)},type:left.checked?'log':'linear'}},{y2}legend:{{orientation:'h'}}}});}}
ds.onchange=draw;left.onchange=draw;{'right.onchange=draw;' if dual_axis else ''}draw();"""
    controls = f"Dataset: <select id='dataset'></select> <label><input id='left' type='checkbox'> left logarithmic</label> {right}"
    _write(os.path.join(output_dir, filename), _shell(title, payload, controls, script))


def _history(output_dir, filename, title, rows, kind):
    payload = [[r["dataset"], r["token_id"], r["token_text_escaped"], r["iteration"],
                r["train_loss"], r["val_loss"], r["training_seen_count"], r["vector_magnitude"],
                r["min_pairwise_angle_deg"]]
               for r in rows]
    has_right = kind == "iteration"
    right = "<label><input id='right' type='checkbox'> right logarithmic</label>" if has_right else ""
    if kind in ("vector", "angle"):
        value_index = 7 if kind == "vector" else 8
        trace_code = f"traces.push({{x:d.map(r=>r[3]),y:d.map(r=>r[{value_index}]),name,mode:'lines+markers'}});"
        x_title = "training iteration"
        y_title = "L2 vector magnitude" if kind == "vector" else "minimum pairwise angle (degrees)"
    else:
        x_index = 6 if kind == "appearances" else 3
        trace_code = (f"traces.push({{x:d.map(r=>r[{x_index}]),y:d.map(r=>r[5]),name:name+' validation',mode:'lines+markers'}},"
                      f"{{x:d.map(r=>r[{x_index}]),y:d.map(r=>r[4]),name:name+' train',mode:'lines+markers',line:{{dash:'dot'}}}});")
        if has_right:
            trace_code += "traces.push({x:d.map(r=>r[3]),y:d.map(r=>r[6]),name:name+' appearances',mode:'lines+markers',yaxis:'y2'});"
        x_title, y_title = ("cumulative appearances" if kind == "appearances" else "training iteration"), "cross-entropy loss"
    y2 = "yaxis2:{title:'cumulative appearances',type:right.checked?'log':'linear',overlaying:'y',side:'right'}," if has_right else ""
    script = f"""const ds=document.getElementById('dataset'),tokens=document.getElementById('tokens'),left=document.getElementById('left'){",right=document.getElementById('right')" if has_right else ""};new Set(rows.map(r=>r[0])).forEach(x=>ds.add(new Option(x,x)));
function fill(){{tokens.replaceChildren();const unique=new Map();rows.filter(r=>r[0]===ds.value).forEach(r=>unique.set(r[1],r[2]));unique.forEach((text,id)=>tokens.add(new Option(`token ${{id}} '${{text}}'`,id)));for(let i=0;i<Math.min(5,tokens.length);i++)tokens.options[i].selected=true;draw();}}
function draw(){{const ids=new Set([...tokens.selectedOptions].map(o=>Number(o.value))),traces=[];ids.forEach(id=>{{const d=rows.filter(r=>r[0]===ds.value&&r[1]===id).sort((a,b)=>a[3]-b[3]),name=`token ${{id}} '${{d.length?d[0][2]:''}}'`;{trace_code}}});plot(traces,{{title:{_json(title)},xaxis:{{title:{_json(x_title)}}},yaxis:{{title:{_json(y_title)},type:left.checked?'log':'linear'}},{y2}legend:{{orientation:'h'}}}});}}
ds.onchange=fill;tokens.onchange=draw;left.onchange=draw;{'right.onchange=draw;' if has_right else ''}fill();"""
    controls = f"Dataset: <select id='dataset'></select> <label><input id='left' type='checkbox'> left logarithmic</label> {right}<br><select id='tokens' multiple size='12'></select>"
    _write(os.path.join(output_dir, filename), _shell(title, payload, controls, script))


def write_per_token_pages(output_dir, rows, summaries, iteration):
    """Write an index plus isolated pages with only the data each graph needs."""
    # Import the static renderer only when the explicitly enabled reporter
    # reaches an export step, keeping report-only code off the startup path.
    from utils.per_token_static import write_static_dashboards

    latest = _latest(rows)
    pages = [
        ("per_token_validation_loss.html", "Validation loss", "val_loss", True, True),
        ("per_token_training_loss.html", "Sampled training loss", "train_loss", True, True),
        ("per_token_training_occurrences.html", "Training occurrences", "training_seen_count", False, True),
        ("per_token_vector_magnitude.html", "Token vector magnitude", "vector_magnitude", True, False),
        ("per_token_min_pairwise_angle.html", "Minimum pairwise angle (degrees)", "min_pairwise_angle_deg", True, False),
    ]
    for filename, title, metric, descending, dual in pages:
        _overview(output_dir, filename, f"{title} at iteration {iteration}", latest, metric, descending, dual)
    histories = [
        ("per_token_loss_by_iteration.html", "Selected-token loss and appearances vs iteration", "iteration"),
        ("per_token_loss_by_appearances.html", "Selected-token loss vs cumulative appearances", "appearances"),
        ("per_token_vector_magnitude_by_iteration.html", "Selected-token vector magnitude vs iteration", "vector"),
        ("per_token_min_pairwise_angle_by_iteration.html", "Selected-token minimum pairwise angle vs iteration", "angle"),
    ]
    for filename, title, kind in histories:
        _history(output_dir, filename, title, rows, kind)
    png_paths = write_static_dashboards(output_dir, rows)
    _write_static_slideshow(output_dir, rows)
    filenames = [p[0] for p in pages] + [p[0] for p in histories] + ["per_token_static_slideshow.html"]
    links = "".join(f"<li><a href='{name}'>{html.escape(name)}</a></li>" for name in filenames)
    png_links = "".join(f"<li><a href='{os.path.basename(path)}'>{html.escape(os.path.basename(path))}</a></li>" for path in png_paths)
    fields = ("dataset", "metric", "populated_tokens", "vocab_size", "mean", "median", "std", "skew", "excess_kurtosis", "min", "max", "p10", "p90", "coefficient_of_variation")
    table = "<table border='1'><tr>" + "".join(f"<th>{f}</th>" for f in fields) + "</tr>" + "".join("<tr>" + "".join(f"<td>{s.get(f, '')}</td>" for f in fields) + "</tr>" for s in summaries) + "</table>"
    _write(os.path.join(output_dir, "per_token_metrics.html"), f"<!doctype html><meta charset='utf-8'><title>Per-token metrics</title><h1>Per-token metrics</h1><h2>Interactive graphs</h2><ul>{links}</ul><h2>Static PNG dashboards</h2><ul>{png_links}</ul><h2>Summary statistics</h2>{table}")
