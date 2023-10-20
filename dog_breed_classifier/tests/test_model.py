# tests/test_model.py

import unittest
import torch
<<<<<<< HEAD
from models.model import PatchEmbedding, ViT

=======
#from models.model import PatchEmbedding, ViT
>>>>>>> dad2c1da0521503b8c09c08b2cce9b51e8d1b8a7

class TestPatchEmbedding(unittest.TestCase):
    def setUp(self):
        self.patch_embedding = PatchEmbedding(image_size=28, patch_size=7)
        self.input = torch.randn(1, 3, 28, 28)

    def test_forward(self):
        output = self.patch_embedding(self.input)
        self.assertEqual(output.size(1), self.patch_embedding.num_patches)
        self.assertEqual(output.size(2), self.patch_embedding.dim)

<<<<<<< HEAD

=======
>>>>>>> dad2c1da0521503b8c09c08b2cce9b51e8d1b8a7
class TestViT(unittest.TestCase):
    def setUp(self):
        self.vit = ViT(
            image_size=28,
            patch_size=7,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
<<<<<<< HEAD
            mlp_dim=2048,
=======
            mlp_dim=2048
>>>>>>> dad2c1da0521503b8c09c08b2cce9b51e8d1b8a7
        )
        self.input = torch.randn(1, 3, 28, 28)

    def test_forward(self):
        output = self.vit(self.input)
        self.assertEqual(output.size(1), self.vit.num_classes)

<<<<<<< HEAD

if __name__ == "__main__":
=======
if __name__ == '__main__':
>>>>>>> dad2c1da0521503b8c09c08b2cce9b51e8d1b8a7
    unittest.main()
