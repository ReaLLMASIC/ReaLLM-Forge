import json
import argparse
from pyvis.network import Network

#TODO: let this take in a meta.pkl file, and create a file to get input to create maps for all initialized vocabulary.

# Argument parser
parser = argparse.ArgumentParser(description="Visualize token probabilities with PyVis.")
parser.add_argument('--threshold', type=float, default=0.1, help='Probability threshold for drawing edges. Default is 0.1.')
parser.add_argument('--remove_from', type=str, help='Character to remove from the network. Use " " for space and "\\n" for newline.')
parser.add_argument('--remove_to', type=str, help='Character to remove from the network. Use " " for space and "\\n" for newline.')
args = parser.parse_args()

# Load the data from stats.json
with open('stats.json', 'r') as f:
    data = json.load(f)

# Function to get color based on character type
def get_color(char):
    if char in 'aeiou':
        return '#FF00FF'  # Light magenta
    elif char in 'bcdfghjklmnpqrstvwxyz5':
        return '#1E90FF'  # Darker blue
    elif char in ['\n', '_', ' ']:
        return '#A9A9A9'  # Dark grey for other characters
    else:
        return '#A9A9A9'  # Dark grey for other characters

# Function to get label for special characters
def get_label(char):
    if char == '\n':
        return '\\n'
    elif char == ' ':
        return 'space'
    elif char == '_':
        return '_'
    else:
        return char

# Create a network
net = Network(directed=True, height="1000px", width="100%", bgcolor="#222222", font_color="white")

# Enable physics with custom settings and controls
net.toggle_physics(True)
net.show_buttons(filter_=['physics'])
net.repulsion(
    spring_length=200,
    spring_strength=0.01,
    damping=0.09,
)

remove_from_char_list = ""
if args.remove_from:
    remove_from_char_list = args.remove_from.replace("\n", "\n")
print(remove_from_char_list)

remove_to_char_list = ""
if args.remove_to:
    remove_to_char_list = args.remove_to.replace("\\n", "\n")
print(remove_to_char_list)

# Add nodes and edges with default threshold
for item in data:
    start_token = item['start_token']
    start_token_label = get_label(start_token)
    token_probs = item['token_probs']
    start_color = get_color(start_token)

    # Check if the character should be removed
    if start_token in remove_from_char_list:
        print("continuing")
        continue

    # Add the start token node if it doesn't exist
    if start_token_label not in net.node_ids:
        net.add_node(
            start_token_label,
            label=start_token_label,
            color=start_color,
            font={'size': 80, 'vadjust': -30},
            shape='circle',
            size=50
        )

    for token, prob in token_probs.items():
        if prob >= args.threshold:
            # Check if the character should be removed
            if token in remove_to_char_list:
                continue
            token_label = get_label(token)
            color = get_color(token)

            # Add the target token node if it doesn't exist
            if token_label not in net.node_ids:
                net.add_node(
                    token_label,
                    label=token_label,
                    color=color,
                    font={'size': 80, 'vadjust': -30},
                    shape='circle',
                    size=50
                )

            # Add an edge with the probability as weight
            net.add_edge(start_token_label, token_label, value=prob, title=str(prob))

# Save the network to an HTML file
net.save_graph('token_probabilities.html')

# Read the generated HTML file
with open('token_probabilities.html', 'r') as file:
    html_content = file.read()

# Add a slider for dynamic threshold adjustment
slider_html = '''
<div style="padding: 20px;">
    <label for="thresholdSlider">Threshold: </label>
    <input type="range" id="thresholdSlider" min="0" max="1" step="0.01" value="0.1" oninput="updateThreshold(this.value)">
    <span id="thresholdValue">0.1</span>
</div>
<script type="text/javascript">
    function updateThreshold(value) {
        document.getElementById("thresholdValue").innerText = value;
        var threshold = parseFloat(value);
        var edges = network.body.data.edges.get();
        edges.forEach(function(edge) {
            if (parseFloat(edge.title) < threshold) {
                edge.hidden = true;
            } else {
                edge.hidden = false;
            }
        });
        network.body.data.edges.update(edges);
    }
</script>
'''

# Inject the slider HTML and JavaScript into the existing HTML content
html_content = html_content.replace('<body>', f'<body>{slider_html}')

# Save the updated HTML content to the same file
with open('token_probabilities.html', 'w') as file:
    file.write(html_content)

