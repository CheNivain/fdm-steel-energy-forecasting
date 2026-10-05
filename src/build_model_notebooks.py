"""Generate unexecuted model notebooks from the readable percent-format scripts."""

from pathlib import Path
import nbformat


ROOT = Path(__file__).resolve().parents[1]


def build_notebook(folder, filename):
    source = (ROOT / folder / "model.py").read_text(encoding="utf-8")
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
        "language_info": {"name": "python"},
    })
    nbformat.validate(notebook)
    path = ROOT / folder / filename
    nbformat.write(notebook, path)
    print(f"Created unexecuted notebook: {path}")


if __name__ == "__main__":
    build_notebook("linear regression", "linear_regression.ipynb")
    build_notebook("random forest", "random_forest.ipynb")
