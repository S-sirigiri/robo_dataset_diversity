import torch
import numpy as np
from transformers import CLIPModel, CLIPImageProcessor, AutoModel, AutoImageProcessor
from PIL import Image
from typing import Union, List

class ImageEmbedder:
    """
    A class to compute image embeddings using CLIP or DINOv2 in batches.

    Args:
        embedding (str): 'clip' or 'dinov2'.
        device (torch.device, optional): torch device to run on (cpu/cuda). Defaults to auto-detect.
        batch_size (int, optional): maximum number of images to process per batch.
    """
    def __init__(self, embedding: str = "clip", device: torch.device = None, batch_size: int = None):
        # Set device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.batch_size = batch_size

        # Load model and processor
        if embedding == "clip":
            model_id = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.processor = CLIPImageProcessor.from_pretrained(model_id)
            self.model_type = "clip"
        elif embedding == "dinov2":
            model_id = "facebook/dinov2-base"
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model_type = "dinov2"
        else:
            raise ValueError(f"Unknown embedding: {embedding}")

    def embed(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        batch_size: int = None
    ) -> torch.Tensor:
        """
        Compute normalized embeddings for one or more images.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): single image at shape (H, W, C) or list of such images.
            batch_size (int, optional): override the default batch_size for this call.

        Returns:
            torch.Tensor: embeddings of shape (N, D), where D is 512 (CLIP) or 768 (DINOv2).
        """
        # Ensure list of images
        if isinstance(images, np.ndarray):
            #images = [images]
            images = list(images)
            #raise ValueError("Provided images must be a numpy array")
        total = len(images)

        # Determine batch size
        bsize = batch_size or self.batch_size or total
        embeddings = []

        # Process in chunks
        for i in range(0, total, bsize):
            batch_imgs = images[i:i + bsize]
            # Convert numpy arrays to PIL images
            pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img
                          for img in batch_imgs]

            # Preprocess and move to device
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                if self.model_type == "clip":
                    feats = self.model.get_image_features(**inputs)
                else:
                    out = self.model(**inputs)
                    feats = out.last_hidden_state[:, 0, :]

            # Normalize embeddings
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(feats.cpu())

        # Concatenate all batches
        return torch.cat(embeddings, dim=0)
