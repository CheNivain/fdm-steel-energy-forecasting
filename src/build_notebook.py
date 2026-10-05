"""Turn the readable percent-format EDA script into an executed notebook."""
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache" / "jupyter"))
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
source = (root / "src" / "eda.py").read_text(encoding="utf-8")
cells = []
for block in source.split("# %%"):
    if not block.strip():
        continue
    if block.startswith(" [markdown]"):
        lines = block.splitlines()[1:]
        markdown = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line for line in lines).strip()
        cells.append(nbformat.v4.new_markdown_cell(markdown))
    else:
        cells.append(nbformat.v4.new_code_cell(block.strip()))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": sys.version.split()[0]},
})
path = root / "notebooks" / "Steel_Industry_EDA.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, path)
client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}})
manager = client.create_kernel_manager()
client.km = manager
# Use the same environment that contains the analysis dependencies.
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
errors = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs if output.output_type == "error"]
if errors:
    raise RuntimeError(errors)
print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells without errors: {path}")
