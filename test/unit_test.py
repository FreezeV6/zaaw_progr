import unittest
from src.utils.image_utils import yolo_to_box
from src.utils.eval import calculate_iou, calculate_accuracy

class TestUtils(unittest.TestCase):
    def yolo_to_box_test(self):
        box = yolo_to_box(0, 0, 100, 50, 200, 100)
        self.assertAlmostEqual(box[0], 0.25)
        self.assertAlmostEqual(box[1], 0.25)
        self.assertAlmostEqual(box[2], 0.5)
        self.assertAlmostEqual(box[3], 0.5)

    def test_iou(self):
        boxA = (0.5, 0.5, 0.2, 0.2)
        boxB = (0.5, 0.5, 0.2, 0.2)
        self.assertAlmostEqual(calculate_iou(boxA, boxB), 1.0)

    def test_accuracy(self):
        self.assertAlmostEqual(calculate_accuracy(['ABC', 'DEF'], ['ABC', 'XXX']), 50.0)

if __name__ == '__main__':
    unittest.main()
