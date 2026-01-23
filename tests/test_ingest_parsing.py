import unittest

from tkg_rag.ingest import parse_extraction_output, parse_timestamp_range, TimestampRange


class TestParseExtractionOutput(unittest.TestCase):
    def test_parses_entities_and_relations(self) -> None:
        tuple_delimiter = "|"
        record_delimiter = ";;"
        raw = (
            f'("entity"{tuple_delimiter}"2021-01-01 to 2021-03-31"{tuple_delimiter}"quarter"){record_delimiter}'
            f'("entity"{tuple_delimiter}"Acme Corp"{tuple_delimiter}"company"){record_delimiter}'
            f'("relationship"{tuple_delimiter}"2021-01-01 to 2021-03-31"{tuple_delimiter}"Acme Corp"{tuple_delimiter}"Beta LLC"{tuple_delimiter}"Acme acquired Beta.")'
        )

        entities, relations = parse_extraction_output(raw, tuple_delimiter, record_delimiter)

        self.assertEqual(2, len(entities))
        self.assertEqual("2021-01-01 to 2021-03-31", entities[0].name)
        self.assertEqual("quarter", entities[0].entity_type)
        self.assertEqual("Acme Corp", entities[1].name)
        self.assertEqual("company", entities[1].entity_type)

        self.assertEqual(1, len(relations))
        rel = relations[0]
        self.assertEqual("2021-01-01 to 2021-03-31", rel.timestamp_entity)
        self.assertEqual("Acme Corp", rel.source_entity)
        self.assertEqual("Beta LLC", rel.target_entity)
        self.assertEqual("Acme acquired Beta.", rel.description)

    def test_ignores_empty_records(self) -> None:
        tuple_delimiter = "|"
        record_delimiter = ";;"
        raw = f'  {record_delimiter}  ("entity"{tuple_delimiter}"Foo"{tuple_delimiter}"company")  {record_delimiter}  '

        entities, relations = parse_extraction_output(raw, tuple_delimiter, record_delimiter)

        self.assertEqual(1, len(entities))
        self.assertEqual(0, len(relations))
        self.assertEqual("Foo", entities[0].name)

    def test_accepts_event_label(self) -> None:
        tuple_delimiter = "|"
        record_delimiter = ";;"
        raw = (
            f'("event"{tuple_delimiter}"2021-05-01 to 2021-05-31"{tuple_delimiter}"Foo"{tuple_delimiter}"Bar"{tuple_delimiter}"Foo announced Bar.")'
        )

        entities, relations = parse_extraction_output(raw, tuple_delimiter, record_delimiter)

        self.assertEqual(0, len(entities))
        self.assertEqual(1, len(relations))
        self.assertEqual("Foo announced Bar.", relations[0].description)


class TestParseTimestampRange(unittest.TestCase):
    def test_parses_exact_date(self) -> None:
        self.assertEqual(
            TimestampRange("2024-02-29", "2024-02-29"),
            parse_timestamp_range("2024-02-29"),
        )

    def test_parses_year(self) -> None:
        self.assertEqual(
            TimestampRange("2023-01-01", "2023-12-31"),
            parse_timestamp_range("2023"),
        )

    def test_parses_quarter_prefix(self) -> None:
        self.assertEqual(
            TimestampRange("2021-01-01", "2021-03-31"),
            parse_timestamp_range("Q1 2021"),
        )

    def test_parses_quarter_suffix(self) -> None:
        self.assertEqual(
            TimestampRange("2022-10-01", "2022-12-31"),
            parse_timestamp_range("2022-Q4"),
        )

    def test_parses_quarter_with_extra_spaces(self) -> None:
        self.assertEqual(
            TimestampRange("1999-10-01", "1999-12-31"),
            parse_timestamp_range("  Q4   1999  "),
        )

    def test_rejects_invalid_quarter(self) -> None:
        self.assertEqual(
            TimestampRange(None, None),
            parse_timestamp_range("2021-Q0"),
        )

    def test_rejects_unsupported_format(self) -> None:
        self.assertEqual(
            TimestampRange(None, None),
            parse_timestamp_range("2021 Q1"),
        )

    def test_unrecognized_date_like_returns_as_is(self) -> None:
        self.assertEqual(
            TimestampRange("2021-13-01", "2021-13-01"),
            parse_timestamp_range("2021-13-01"),
        )

    def test_empty_string_returns_none(self) -> None:
        self.assertEqual(
            TimestampRange(None, None),
            parse_timestamp_range(""),
        )

    def test_unrecognized_returns_none(self) -> None:
        self.assertEqual(
            TimestampRange(None, None),
            parse_timestamp_range("last summer"),
        )


if __name__ == "__main__":
    unittest.main()
