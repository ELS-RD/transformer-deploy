from pathlib import Path

import mkdocs_gen_files


skip_list = ["init", "python_tokenizer"]

src_root = Path("src/transformer_deploy")
for path in src_root.glob("**/*.py"):
    if any([s in str(path) for s in skip_list]):
        continue
    doc_path = Path("reference", path.relative_to(src_root)).with_suffix(".md")

    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = ".".join(path.with_suffix("").parts)
        print("::: " + ident, file=f)
