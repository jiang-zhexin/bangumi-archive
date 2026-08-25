import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

dir = Path("data")


def main():
    archive_path = download()
    transformer(archive_path=archive_path, output_dir=dir)


def download() -> Path:
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/bangumi/Archive/refs/heads/master/aux/latest.json"
    ) as resp:
        data = resp.read().decode("utf-8")
        aux = json.loads(data)
        download_url: str = aux["browser_download_url"]
        path = Path(aux["name"])

    if not path.exists():
        print("start downloading...")
        with (
            urllib.request.urlopen(download_url) as resp,
            open(path, "wb") as out_file,
        ):
            shutil.copyfileobj(resp, out_file)
            print("download success!")
    return path


def transformer(archive_path: Path, output_dir: Path, compression_level: int = 9):
    output_dir.mkdir(parents=True, exist_ok=True)

    repace_str = pl.col(pl.String).replace("", None)

    with zipfile.ZipFile(archive_path) as f:
        with f.open("character.jsonlines") as character:
            character_df = pl.scan_ndjson(character).with_columns(repace_str)
        character_df.sink_parquet(
            output_dir / "character.parquet", compression_level=compression_level
        )

        with f.open("episode.jsonlines") as episode:
            episode_df = (
                pl.scan_ndjson(episode, ignore_errors=True)
                .with_columns(repace_str)
                .with_columns(pl.col("airdate").str.to_date("%Y-%m-%d", strict=False))
            )
        episode_df.sink_parquet(
            output_dir / "episode.parquet", compression_level=compression_level
        )

        with f.open("person-characters.jsonlines") as person_characters:
            person_characters_df = pl.scan_ndjson(person_characters).with_columns(
                repace_str
            )
        person_characters_df.sink_parquet(
            output_dir / "person-characters.parquet",
            compression_level=compression_level,
        )

        with f.open("person-relations.jsonlines") as person_relations:
            person_relations_df = pl.scan_ndjson(
                person_relations,
                schema_overrides={"person_type": pl.Enum(("prsn", "crt"))},
            )
        person_relations_df.sink_parquet(
            output_dir / "person-relations.parquet", compression_level=compression_level
        )

        with f.open("person.jsonlines") as person:
            person_df = pl.scan_ndjson(person).with_columns(repace_str)
        person_df.sink_parquet(
            output_dir / "person.parquet", compression_level=compression_level
        )

        with f.open("subject-characters.jsonlines") as subject_characters:
            subject_characters_df = pl.scan_ndjson(subject_characters)
        subject_characters_df.sink_parquet(
            output_dir / "subject-characters.parquet",
            compression_level=compression_level,
        )

        with f.open("subject-persons.jsonlines") as subject_persons:
            subject_persons_df = pl.scan_ndjson(subject_persons).with_columns(
                repace_str
            )
        subject_persons_df.sink_parquet(
            output_dir / "subject-persons.parquet", compression_level=compression_level
        )

        with f.open("subject-relations.jsonlines") as subject_relations:
            subject_relations_df = pl.scan_ndjson(subject_relations)
        subject_relations_df.sink_parquet(
            output_dir / "subject-relations.parquet",
            compression_level=compression_level,
        )

        with f.open("subject.jsonlines") as subject:
            subject_df = (
                pl.scan_ndjson(subject)
                .with_columns(repace_str)
                .with_columns(pl.col("date").str.to_date("%Y-%m-%d", strict=False))
            )
        subject_df.sink_parquet(
            output_dir / "subject.parquet", compression_level=compression_level
        )


if __name__ == "__main__":
    main()
