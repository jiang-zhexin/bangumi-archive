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
            character_df = pl.scan_ndjson(
                character,
                schema={
                    "id": pl.UInt64,
                    "role": pl.UInt32,
                    "name": pl.String,
                    "infobox": pl.String,
                    "summary": pl.String,
                    "comments": pl.UInt32,
                    "collects": pl.UInt32,
                },
            ).with_columns(repace_str)
        character_df.sink_parquet(
            output_dir / "character.parquet",
            compression_level=compression_level,
        )

        with f.open("episode.jsonlines") as episode:
            episode_df = pl.scan_ndjson(
                episode,
                ignore_errors=True,
                schema={
                    "id": pl.UInt64,
                    "name": pl.String,
                    "name_cn": pl.String,
                    "description": pl.String,
                    "airdate": pl.Date,
                    "disc": pl.UInt32,
                    "duration": pl.String,
                    "subject_id": pl.UInt64,
                    "sort": pl.UInt32,
                    "type": pl.UInt32,
                },
            ).with_columns(repace_str)
        episode_df.sink_parquet(
            output_dir / "episode.parquet",
            compression_level=compression_level,
        )

        with f.open("person-characters.jsonlines") as person_characters:
            person_characters_df = pl.scan_ndjson(
                person_characters,
                schema={
                    "person_id": pl.UInt64,
                    "subject_id": pl.UInt64,
                    "character_id": pl.UInt64,
                    "type": pl.UInt32,
                    "summary": pl.String,
                },
            ).with_columns(repace_str)
        person_characters_df.sink_parquet(
            output_dir / "person-characters.parquet",
            compression_level=compression_level,
        )

        with f.open("person-relations.jsonlines") as person_relations:
            person_relations_df = pl.scan_ndjson(
                person_relations,
                schema={
                    "person_type": pl.Enum(("prsn", "crt")),
                    "person_id": pl.UInt64,
                    "related_person_id": pl.UInt64,
                    "relation_type": pl.UInt32,
                    "spoiler": pl.Boolean,
                    "ended": pl.Boolean,
                },
            )
        person_relations_df.sink_parquet(
            output_dir / "person-relations.parquet",
            compression_level=compression_level,
        )

        with f.open("person.jsonlines") as person:
            person_df = pl.scan_ndjson(
                person,
                schema={
                    "id": pl.UInt64,
                    "name": pl.String,
                    "type": pl.UInt32,
                    "career": pl.List(pl.String),
                    "infobox": pl.String,
                    "summary": pl.String,
                    "comments": pl.UInt32,
                    "collects": pl.UInt32,
                },
            ).with_columns(repace_str)
        person_df.sink_parquet(
            output_dir / "person.parquet",
            compression_level=compression_level,
        )

        with f.open("subject-characters.jsonlines") as subject_characters:
            subject_characters_df = pl.scan_ndjson(
                subject_characters,
                schema={
                    "character_id": pl.UInt64,
                    "subject_id": pl.UInt64,
                    "type": pl.UInt32,
                    "order": pl.UInt32,
                },
            )
        subject_characters_df.sink_parquet(
            output_dir / "subject-characters.parquet",
            compression_level=compression_level,
        )

        with f.open("subject-persons.jsonlines") as subject_persons:
            subject_persons_df = pl.scan_ndjson(
                subject_persons,
                schema={
                    "person_id": pl.UInt64,
                    "subject_id": pl.UInt64,
                    "position": pl.UInt32,
                    "appear_eps": pl.String,
                },
            ).with_columns(repace_str)
        subject_persons_df.sink_parquet(
            output_dir / "subject-persons.parquet",
            compression_level=compression_level,
        )

        with f.open("subject-relations.jsonlines") as subject_relations:
            subject_relations_df = pl.scan_ndjson(
                subject_relations,
                schema={
                    "subject_id": pl.UInt64,
                    "relation_type": pl.UInt32,
                    "related_subject_id": pl.UInt64,
                    "order": pl.UInt32,
                },
            )
        subject_relations_df.sink_parquet(
            output_dir / "subject-relations.parquet",
            compression_level=compression_level,
        )

        with f.open("subject.jsonlines") as subject:
            subject_df = pl.scan_ndjson(
                subject,
                ignore_errors=True,
                schema={
                    "id": pl.UInt64,
                    "type": pl.UInt32,
                    "name": pl.String,
                    "name_cn": pl.String,
                    "infobox": pl.String,
                    "platform": pl.UInt32,
                    "summary": pl.String,
                    "nsfw": pl.Boolean,
                    "tags": pl.List(
                        pl.Struct(
                            {
                                "name": pl.String,
                                "count": pl.UInt32,
                            }
                        )
                    ),
                    "meta_tags": pl.List(pl.String),
                    "score": pl.Float64,
                    "score_details": pl.Struct(
                        {f"{i}": pl.UInt32 for i in range(1, 11)}
                    ),
                    "rank": pl.UInt32,
                    "date": pl.Date,
                    "favorite": pl.Struct(
                        {
                            "wish": pl.UInt32,
                            "done": pl.UInt32,
                            "doing": pl.UInt32,
                            "on_hold": pl.UInt32,
                            "dropped": pl.UInt32,
                        }
                    ),
                    "series": pl.Boolean,
                },
            ).with_columns(repace_str)
        subject_df.sink_parquet(
            output_dir / "subject.parquet",
            compression_level=compression_level,
        )


if __name__ == "__main__":
    main()
