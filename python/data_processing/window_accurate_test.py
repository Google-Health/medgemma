import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from data_processing import image_utils

class TestWindowAccurate(parameterized.TestCase):
  """Unit tests for `window_accurate()`."""

  @parameterized.named_parameters(
      ('ExactCenter', 2048, 2048, 4096, 32768), # Midpoint (ceil of 32767.5)
      ('BottomEdge', 0, 2048, 4096, 0),        # Bottom clip
      ('TopEdge', 4096, 2048, 4096, 65535),    # Top clip
      ('BelowBottom', -100, 2048, 4096, 0),    # Below bottom clip
      ('AboveTop', 5000, 2048, 4096, 65535),   # Above top clip
  )
  def testStandardRange(self, input_value: int, center: int, width: int, expected: int):
    """Tests standard 12-bit range with uint16 output."""
    actual = image_utils.window_accurate(
        np.array([input_value], dtype=np.int16),
        center,
        width,
        np.uint16
    )
    self.assertEqual(actual[0], expected)

  @parameterized.named_parameters(
      ('BeforeLowest', 2045, 2048, 4, 0),      # Center=2048, Width=4 -> [2046, 2050]
      ('AtLowest', 2046, 2048, 4, 0),          # At bottom clip
      ('Midway', 2048, 2048, 4, 32768),        # At center
      ('AtHighest', 2050, 2048, 4, 65535),      # At top clip
      ('AfterHighest', 2051, 2048, 4, 65535),  # After top clip
  )
  def testSmallWindow(self, input_value: int, center: int, width: int, expected: int):
    """Tests behavior with a very narrow window."""
    actual = image_utils.window_accurate(
        np.array([input_value], dtype=np.int16),
        center,
        width,
        np.uint16
    )
    self.assertEqual(actual[0], expected)

  def testRoundingCorrectness(self):
    """Specifically tests that rounding to nearest integer is working."""
    # Window 100 to 200 (Center 150, Width 100)
    # Norm to 0-255 (uint8)
    image = np.array([125], dtype=np.int16) # Exactly 1/4 of the way: (125-100)/(200-100) * 255 = 0.25 * 255 = 63.75
    # Round(63.75) should be 64.
    actual = image_utils.window_accurate(image, 150, 100, np.uint8)
    self.assertEqual(actual[0], 64)

    image_low = np.array([124], dtype=np.int16) # (124-100)/100 * 255 = 0.24 * 255 = 61.2
    # Round(61.2) should be 61.
    actual_low = image_utils.window_accurate(image_low, 150, 100, np.uint8)
    self.assertEqual(actual_low[0], 61)

if __name__ == '__main__':
  absltest.main()
