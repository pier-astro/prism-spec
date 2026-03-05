"""
CSS styles for model display tables.

Centralized styling for consistent appearance across 
terminal and Jupyter notebook outputs.
"""

# Model summary table CSS
MODEL_TABLE_CSS = """
.fantasylab-model-summary {
  border-collapse: collapse;
  font-family: Arial, sans-serif;
  font-size: 13px;
  margin-top: 10px;
}
.fantasylab-model-expr {
  font-family: "Courier New", Consolas, monospace;
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 8px;
  color: #888;
}
.fantasylab-model-summary th {
  background-color: #e8e8e8;
  border: 1px solid #ccc;
  padding: 8px 10px;
  text-align: left;
  font-weight: bold;
  color: #333;
}
.fantasylab-model-summary td {
  border: 1px solid #ddd;
  padding: 6px 10px;
  color: #333;
}
.fantasylab-model-summary tr:nth-child(even) {
  background-color: #f5f5f5;
}
.fantasylab-model-summary tr:nth-child(odd) {
  background-color: #ffffff;
}
.fantasylab-model-summary .component-name {
  font-family: "Courier New", Consolas, monospace;
  color: #333;
}
.fantasylab-model-summary .param-name {
  font-family: "Courier New", Consolas, monospace;
  color: #0066cc;
}
.fantasylab-model-summary .constraint {
  color: #666;
  font-style: italic;
  font-size: 12px;
}
.fantasylab-model-summary .uncertainty {
  font-size: 12px;
}
.fantasylab-model-summary .free-yes {
  color: #00aa00;
  text-align: center;
  font-size: 18px;
}
.fantasylab-model-summary .free-no {
  color: #cc0000;
  text-align: center;
  font-size: 18px;
}
"""

# Collapsible table CSS
COLLAPSIBLE_CSS = """
.fantasylab-toggle-btn {
  background-color: #007acc;
  color: white;
  border: none;
  padding: 6px 12px;
  cursor: pointer;
  border-radius: 4px;
  font-size: 13px;
  margin-bottom: 8px;
}
.fantasylab-toggle-btn:hover {
  background-color: #005a9e;
}
.fantasylab-collapsible-table {
  display: none;
}
.fantasylab-collapsible-table.expanded {
  display: table;
}
"""

# Flux table CSS (simpler version)
FLUX_TABLE_CSS = """
.fantasylab-flux-table {
  border-collapse: collapse;
  font-family: Arial, sans-serif;
  font-size: 13px;
}
.fantasylab-flux-table th {
  background-color: #e8e8e8;
  border: 1px solid #ccc;
  padding: 8px 10px;
  text-align: left;
}
.fantasylab-flux-table td {
  border: 1px solid #ddd;
  padding: 6px 10px;
}
"""

# Toggle JavaScript (generated per table)
def get_toggle_script(table_id, num_params):
    """Generate JavaScript for collapsible table toggle."""
    return f"""
function toggleTable_{table_id}() {{
  var table = document.getElementById("{table_id}");
  var btn = document.getElementById("btn_{table_id}");
  if (table.classList.contains("expanded")) {{
    table.classList.remove("expanded");
    btn.innerHTML = "▼ Show All {num_params} Parameters";
  }} else {{
    table.classList.add("expanded");
    btn.innerHTML = "▲ Hide Parameters";
  }}
}}
"""
