from pathlib import Path

parts = Path.cwd().parts
parts = parts[:parts.index("reagents")]
root = Path(*parts) / "reagents"
translate_script_path = str(root / "translate.py")
