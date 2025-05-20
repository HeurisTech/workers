from jinja2 import Environment, BaseLoader
import json

# Your input JSON (like the one you showed earlier)
with open('j2_input.json', 'r') as file:
    input_data = json.load(file)

# Let's read the jinja template from a file
with open('j2_lm_default_template.j2', 'r') as file:
    jinja_template = file.read()

# Set up a Jinja environment
env = Environment(loader=BaseLoader())
env.filters['tojson'] = lambda value: json.dumps(value, indent=4)  # add tojson filter
template = env.from_string(jinja_template)

# Render it
rendered_output = template.render(**input_data)

# Output the final result
with open('j2_output.txt', 'w') as file:
    file.write(rendered_output)