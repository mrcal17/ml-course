"""Comprehensive test battery for the ML course."""
import ast, glob, re, os, subprocess

os.chdir(os.path.dirname(__file__))

errors = []
warnings = []

print("=" * 60)
print("ML COURSE — COMPREHENSIVE TEST BATTERY")
print("=" * 60)

# TEST 1: All notebooks parse as valid Python
print("\n--- TEST 1: Python syntax validation ---")
notebooks = sorted(glob.glob("notebooks/*.py"))
for nb in notebooks:
    try:
        with open(nb, encoding="utf-8") as f:
            ast.parse(f.read())
        print(f"  OK: {os.path.basename(nb)}")
    except SyntaxError as e:
        errors.append(f"SYNTAX ERROR in {nb}: {e}")
        print(f"  FAIL: {os.path.basename(nb)}: {e}")

# TEST 2: Marimo structure
print("\n--- TEST 2: Marimo structure validation ---")
for nb in notebooks:
    with open(nb, encoding="utf-8") as f:
        content = f.read()
    name = os.path.basename(nb)
    issues = []
    if "import marimo" not in content:
        issues.append("missing import marimo")
    if "marimo.App(" not in content:
        issues.append("missing marimo.App()")
    if "@app.cell" not in content:
        issues.append("no @app.cell decorators")
    if "app.run()" not in content:
        issues.append("missing app.run()")
    if issues:
        errors.extend([f"{name}: {i}" for i in issues])
        print(f"  FAIL: {name} — {', '.join(issues)}")
    else:
        cell_count = content.count("@app.cell")
        print(f"  OK: {name} ({cell_count} cells)")

# TEST 3: No remaining Obsidian wiki-links
print("\n--- TEST 3: No remaining Obsidian wiki-links ---")
wiki_link_pattern = re.compile(r"\[\[textbooks/.*?\]\]")
broken = 0
for nb in notebooks:
    with open(nb, encoding="utf-8") as f:
        content = f.read()
    matches = wiki_link_pattern.findall(content)
    if matches:
        broken += 1
        warnings.append(f"{os.path.basename(nb)}: {len(matches)} unconverted wiki-links")
        print(f"  WARN: {os.path.basename(nb)} has {len(matches)} unconverted [[wiki-links]]")
if broken == 0:
    print("  OK: No remaining wiki-links found")

# TEST 4: Textbook PDFs
print("\n--- TEST 4: Textbook PDFs ---")
expected_textbooks = [
    "MML.pdf", "ISLR.pdf", "ESL.pdf", "Boyd-ConvexOptimization.pdf",
    "DLBook.pdf", "Bishop-PRML.pdf", "Sutton-RL.pdf", "Murphy-PML1.pdf",
    "Murphy-PML2.pdf", "Chan-Probability.pdf", "Geron-HandsOnML.pdf",
    "Wasserman-AllOfStatistics.pdf",
]
for tb in expected_textbooks:
    path = f"textbooks/{tb}"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  OK: {tb} ({size_mb:.1f} MB)")
    else:
        errors.append(f"Missing textbook: {tb}")
        print(f"  FAIL: {tb} — NOT FOUND")

# TEST 5: Animation source files
print("\n--- TEST 5: Animation source files ---")
expected_anim_sources = [
    "calculus.py", "linear_algebra.py", "probability.py", "optimization.py",
    "regression.py", "classification.py", "trees.py", "unsupervised.py",
    "neural_nets.py", "cnn.py", "rnn.py", "attention.py", "generative.py", "rl.py",
]
all_scenes = set()
for src in expected_anim_sources:
    path = f"animations/src/{src}"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        scenes = re.findall(r"class (\w+)\(.*Scene\)", content)
        all_scenes.update(scenes)
        print(f"  OK: {src} — scenes: {scenes}")
    else:
        errors.append(f"Missing animation source: {src}")
        print(f"  FAIL: {src} — NOT FOUND")

# TEST 6: Animation embed references match actual scenes
print("\n--- TEST 6: Animation embed references ---")
embed_pattern = re.compile(r"animations/rendered/(\w+)\.gif")
embed_count = 0
for nb in notebooks:
    with open(nb, encoding="utf-8") as f:
        content = f.read()
    for m in embed_pattern.finditer(content):
        scene_name = m.group(1)
        embed_count += 1
        if scene_name in all_scenes:
            print(f"  OK: {os.path.basename(nb)} -> {scene_name}.gif")
        else:
            errors.append(f"{os.path.basename(nb)} refs non-existent scene: {scene_name}")
            print(f"  FAIL: {os.path.basename(nb)} -> {scene_name}.gif (NOT FOUND)")
print(f"  Total embeds: {embed_count}")

# TEST 7: Rendered GIFs
print("\n--- TEST 7: Rendered GIFs ---")
rendered_gifs = glob.glob("animations/rendered/*.gif")
print(f"  {len(rendered_gifs)} of {len(all_scenes)} animations rendered")
for gif in sorted(rendered_gifs):
    size_kb = os.path.getsize(gif) / 1024
    print(f"  OK: {os.path.basename(gif)} ({size_kb:.0f} KB)")
if len(rendered_gifs) < len(all_scenes):
    warnings.append(f"Only {len(rendered_gifs)}/{len(all_scenes)} animations rendered (rendering in background)")

# TEST 8: Expected notebooks completeness
print("\n--- TEST 8: Expected notebooks completeness ---")
expected = [
    "home.py",
    "0a_python.py", "0b_calculus.py", "0c_linear_algebra.py",
    "0d_probability.py", "0e_estimation.py", "0f_optimization.py",
    "1a_ml_landscape.py", "1b_linear_regression.py", "1c_classification.py",
    "1d_model_selection.py", "1e_trees_ensembles.py", "1f_unsupervised.py",
    "2a_neural_networks.py", "2b_dl_optimization.py", "2c_regularization.py",
    "2d_cnn.py", "2e_sequence_models.py",
    "3a_transformers.py", "3b_generative_models.py", "3c_self_supervised.py",
    "3d_reinforcement_learning.py",
    "4a_nlp.py", "4b_computer_vision.py", "4c_advanced_rl.py", "4d_bayesian_ml.py",
]
missing = [e for e in expected if not os.path.exists(f"notebooks/{e}")]
if missing:
    for m in missing:
        errors.append(f"Missing notebook: {m}")
        print(f"  FAIL: {m}")
else:
    print(f"  OK: All {len(expected)} expected notebooks present")

# TEST 9: Interactive widgets
print("\n--- TEST 9: Interactive widgets ---")
widget_checks = {
    "0b_calculus.py": ["mo.ui.slider"],
    "0d_probability.py": ["mo.ui.dropdown", "mo.ui.slider"],
    "0f_optimization.py": ["mo.ui.slider"],
    "1b_linear_regression.py": ["mo.ui.slider"],
    "1c_classification.py": ["mo.ui.slider"],
    "1e_trees_ensembles.py": ["mo.ui.slider"],
    "1f_unsupervised.py": ["mo.ui.slider"],
    "2a_neural_networks.py": ["mo.ui.slider"],
    "2c_regularization.py": ["mo.ui.slider"],
}
for nb_name, widgets in widget_checks.items():
    path = f"notebooks/{nb_name}"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        found = [w for w in widgets if w in content]
        missing_w = [w for w in widgets if w not in content]
        if missing_w:
            warnings.append(f"{nb_name}: missing widgets {missing_w}")
            print(f"  WARN: {nb_name} — missing: {missing_w}")
        else:
            print(f"  OK: {nb_name} — has {found}")

# TEST 10: Notebook link references in home.py
print("\n--- TEST 10: Home page links ---")
with open("notebooks/home.py", encoding="utf-8") as f:
    home_content = f.read()
link_pattern = re.compile(r"\((\w+\.py)\)")
home_links = link_pattern.findall(home_content)
for link in home_links:
    if os.path.exists(f"notebooks/{link}"):
        print(f"  OK: {link} exists")
    else:
        errors.append(f"Home links to non-existent: {link}")
        print(f"  FAIL: {link} — NOT FOUND")
print(f"  Total links in home: {len(home_links)}")

# TEST 11: Server responsiveness
print("\n--- TEST 11: Marimo server ---")
try:
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:2718/"],
        capture_output=True, text=True, timeout=5,
    )
    code = result.stdout.strip()
    if code in ["200", "303"]:
        print(f"  OK: Server responding (HTTP {code})")
    else:
        warnings.append(f"Server returned HTTP {code}")
        print(f"  WARN: Server returned HTTP {code}")
except Exception as e:
    warnings.append(f"Server check failed: {e}")
    print(f"  WARN: Could not reach server: {e}")

# TEST 12: Notebook file sizes
print("\n--- TEST 12: Notebook sizes (detect truncation) ---")
for nb in notebooks:
    name = os.path.basename(nb)
    size_kb = os.path.getsize(nb) / 1024
    if size_kb < 5 and name != "home.py":
        warnings.append(f"{name}: suspiciously small ({size_kb:.1f} KB)")
        print(f"  WARN: {name} — only {size_kb:.1f} KB")
    else:
        print(f"  OK: {name} — {size_kb:.1f} KB")

# TEST 13: Textbook links in notebooks point to existing files
print("\n--- TEST 13: Textbook link targets ---")
tb_link_pattern = re.compile(r"\.\./textbooks/([\w\-]+\.pdf)")
all_tb_refs = set()
for nb in notebooks:
    with open(nb, encoding="utf-8") as f:
        content = f.read()
    for m in tb_link_pattern.finditer(content):
        all_tb_refs.add(m.group(1))
for ref in sorted(all_tb_refs):
    if os.path.exists(f"textbooks/{ref}"):
        print(f"  OK: {ref} referenced and exists")
    else:
        errors.append(f"Textbook link target missing: {ref}")
        print(f"  FAIL: {ref} referenced but NOT FOUND")
print(f"  Total unique textbook references: {len(all_tb_refs)}")

# SUMMARY
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Errors:   {len(errors)}")
print(f"  Warnings: {len(warnings)}")
if errors:
    print("\n  ERRORS:")
    for e in errors:
        print(f"    - {e}")
if warnings:
    print("\n  WARNINGS:")
    for w in warnings:
        print(f"    - {w}")
if not errors:
    print("\n  ALL CRITICAL TESTS PASSED")
