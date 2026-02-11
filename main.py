import argparse
from ingest.detect import detect_file_type
from ingest.extract import extract_text
from ingest.markdown import to_markdown
from utils.fs import iter_files, safe_write

def main():
    parser = argparse.ArgumentParser(description="Neural ingestion → Obsidian vault")
    parser.add_argument("input", help="Input folder")
    parser.add_argument("--out", default="vault", help="Output Obsidian vault folder")

    args = parser.parse_args()

    for file in iter_files(args.input):
        print("Processing:", file)

        ftype = detect_file_type(file)
        text = extract_text(file, ftype)

        if not text.strip():
            continue

        md = to_markdown(text, source=file)

        print("Writing markdown for:", file)
        safe_write(md, file, args.out)

if __name__ == "__main__":
    main()

