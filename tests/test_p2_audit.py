"""Isolated checks for the source-only P2 auditor; never open target data."""
import unittest

from scripts.audit_p2_source import parse_query, safe_reference_sql, three_valued


class P2ParserTests(unittest.TestCase):
    def test_from_where_and_reversed_bound(self):
        row = parse_query(
            'SELECT COUNT(*) FROM t a, u b WHERE a.x = b.x AND 10 <= a.y AND a.y < 20'
        )
        self.assertEqual(row['tables'], ['a', 'b'])
        self.assertEqual(row['join_columns'], ['a.x', 'b.x'])
        self.assertEqual(row['filter_columns'], ['a.y'])
        self.assertEqual(row['predicates'], ['EQ', 'GTE', 'LT'])
        self.assertEqual(sorted(row['repeated']['a.y']), ['GTE', 'LT'])
        self.assertEqual(row['unsupported'], [])

    def test_or_and_unsupported_remain_visible(self):
        row = parse_query('SELECT COUNT(*) FROM t a WHERE (a.x = 1 OR a.x IN (2,3))')
        self.assertIn('OR', row['boolean_tree'])
        self.assertIn('IN', row['unsupported'])

    def test_ambiguous_unqualified_column_rejected(self):
        row = parse_query('SELECT COUNT(*) FROM t a, u b WHERE x = 2')
        self.assertIn('unqualified_column', row['unsupported'])

    def test_join_on_and_outer_join_not_projected(self):
        row = parse_query('SELECT COUNT(*) FROM t a LEFT JOIN u b ON a.x=b.x WHERE b.y IS NULL')
        self.assertIn('outer_join', row['unsupported'])
        self.assertIn('IS', row['unsupported'])
        self.assertEqual(row['filter_columns'], ['b.y'])

    def test_sql_three_valued_logic(self):
        self.assertIsNone(three_valued('AND', True, None))
        self.assertFalse(three_valued('AND', False, None))
        self.assertTrue(three_valued('OR', True, None))
        self.assertIsNone(three_valued('NOT', None))

    def test_reference_sql_never_uses_unescaped_identifier_or_literal(self):
        sql, args = safe_reference_sql('weird"table', 'x"col', 7)
        self.assertEqual(sql, 'SELECT COUNT(*) FROM "weird""table" WHERE "x""col" = %s')
        self.assertEqual(args, (7,))
        with self.assertRaises(ValueError):
            safe_reference_sql('t;DROP TABLE x', 'x', 1)


if __name__ == '__main__':
    unittest.main()
