import unittest

from tkg_rag.retrieve import format_context


class TestFormatContext(unittest.TestCase):
    def test_empty_items_returns_message(self) -> None:
        self.assertEqual("No matching context found.", format_context([])[0])

    def test_formats_chunk_and_edge(self) -> None:
        items = [
            {
                "kind": "chunk",
                "text": "  Chunk text with spaces.  ",
                "chunk_id": 42,
            },
            {
                "kind": "edge",
                "relation_text": "  Acme acquired Beta.  ",
                "chunk_ids": [42],
                "rel_id": 7,
            },
        ]

        output, _ = format_context(items)
        lines = output.splitlines()

        self.assertTrue(lines[0].startswith("Context from a temporal knowledge graph."))
        self.assertEqual("[c_id:1] Chunk text with spaces.", lines[2])
        self.assertEqual("[e_id:1] Acme acquired Beta.", lines[3])
        self.assertEqual("source_id: 2", lines[4])

    def test_skips_empty_chunk_text(self) -> None:
        items = [
            {
                "kind": "chunk",
                "text": "   ",
                "chunk_id": 1,
            }
        ]

        output,_ = format_context(items)
        lines = output.splitlines()

        self.assertEqual(
            [
                "Context from a temporal knowledge graph. E stands for edge c for chunk with an id[e_id:N] or [c_id:N] ",
            ],
            lines,
        )


if __name__ == "__main__":
    unittest.main()
