"""Export all Marimo notebooks to static HTML for GitHub Pages."""
import subprocess
import os
import glob
import shutil

os.chdir(os.path.dirname(__file__))

DOCS_DIR = "docs"
NOTEBOOKS_DIR = "notebooks"

# Clean and recreate docs
if os.path.exists(DOCS_DIR):
    shutil.rmtree(DOCS_DIR)
os.makedirs(DOCS_DIR, exist_ok=True)

# Copy animations to docs so they're accessible
anim_src = "animations/rendered"
anim_dst = os.path.join(DOCS_DIR, "animations", "rendered")
if os.path.exists(anim_src):
    os.makedirs(anim_dst, exist_ok=True)
    for gif in glob.glob(os.path.join(anim_src, "*.gif")):
        shutil.copy2(gif, anim_dst)
        print(f"  Copied: {os.path.basename(gif)}")

# Export each notebook to HTML
notebooks = sorted(glob.glob(os.path.join(NOTEBOOKS_DIR, "*.py")))

for nb in notebooks:
    name = os.path.basename(nb).replace(".py", "")

    # home.py -> index.html, others -> name.html
    if name == "home":
        out_name = "index.html"
    else:
        out_name = f"{name}.html"

    out_path = os.path.join(DOCS_DIR, out_name)

    print(f"Exporting {name}...")
    result = subprocess.run(
        ["python", "-m", "marimo", "export", "html", nb, "-o", out_path],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode == 0 and os.path.exists(out_path):
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  OK: {out_name} ({size_kb:.0f} KB)")
    else:
        print(f"  FAILED: {out_name}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")

# Fix links in index.html: /?file=X.py -> X.html
index_path = os.path.join(DOCS_DIR, "index.html")
if os.path.exists(index_path):
    with open(index_path, encoding="utf-8") as f:
        content = f.read()

    # Fix module links
    import re
    content = re.sub(
        r'href="/?[?]file=(\w+)\.py"',
        lambda m: f'href="{m.group(1)}.html"',
        content,
    )
    content = re.sub(
        r'href="/\?file=(\w+)\.py"',
        lambda m: f'href="{m.group(1)}.html"',
        content,
    )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("\nFixed links in index.html")

print(f"\nSite built in {DOCS_DIR}/")
print(f"Total files: {len(glob.glob(os.path.join(DOCS_DIR, '*.html')))}")
