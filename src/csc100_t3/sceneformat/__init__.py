from pathlib import Path

import xmlformatter


SCENE_ROOT = Path(__file__).resolve().parents[3] / "scene"


def run():
    paths = sorted(SCENE_ROOT.rglob("*.xml"))

    if not paths:
        raise FileNotFoundError(f"No XML files found in {SCENE_ROOT}")

    formatter = xmlformatter.Formatter(
        indent=4,
        indent_char=" ",
        eof_newline=True,
        selfclose=True,
        preserve_attributes=True,
        blanks=True,
    )

    formatted_files = {}
    for path in paths:
        source = path.read_bytes()
        formatted_files[path] = (source, formatter.format_string(source))

    for path, (source, formatted) in formatted_files.items():
        if source == formatted:
            continue

        print(f"Formatting {path}")
        path.write_bytes(formatted)
