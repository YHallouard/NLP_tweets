import unittest


class TestStringMethods(unittest.TestCase):

    def test_format(self):
        # GIVEN
        tester = 0

        expected_res = 0

        # WHEN
        g = lambda x: x
        res = g(tester)

        # THEN
        self.assertEqual(expected_res, res)
