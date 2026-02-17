"""
run_tests.py
------------
Run ALL unit tests with a single command:
    python run_tests.py
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover all tests in the tests/ directory
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')
    discovered = loader.discover(start_dir=tests_dir, pattern='test_*.py')
    suite.addTests(discovered)

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*60)
    print(f"Tests run   : {result.testsRun}")
    print(f"Failures    : {len(result.failures)}")
    print(f"Errors      : {len(result.errors)}")
    print(f"Skipped     : {len(result.skipped)}")
    print("="*60)

    if result.wasSuccessful():
        print("✅  ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
