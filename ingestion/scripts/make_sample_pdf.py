"""Generate a synthetic paper PDF (title/authors/abstract/sections/refs) for testing."""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import sys

out_path = sys.argv[1] if len(sys.argv) > 1 else "papers/sample_001.pdf"

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, leading=22)
author_style = ParagraphStyle("AuthorX", parent=styles["Normal"], fontSize=11, alignment=1)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10)
body = styles["Normal"]

doc = SimpleDocTemplate(out_path, pagesize=letter)
story = []

story.append(Paragraph("ViTGAN: Generative Adversarial Networks with Vision Transformers", title_style))
story.append(Spacer(1, 8))
story.append(Paragraph("Jane Doe, John Smith, and Alice Wu", author_style))
story.append(Spacer(1, 14))

story.append(Paragraph("Abstract", h1))
story.append(Paragraph(
    "We introduce ViTGAN, a purely transformer-based generative adversarial network for "
    "high-fidelity image synthesis. Our model removes convolutional inductive biases and "
    "demonstrates competitive results on standard benchmarks such as CIFAR-10 and CelebA. "
    "arXiv:2107.04589", body))

story.append(Paragraph("1 Introduction", h1))
story.append(Paragraph(
    "Generative adversarial networks (GANs) have become a dominant paradigm for image "
    "synthesis since their introduction. Recently, vision transformers have shown strong "
    "performance on discriminative tasks, motivating their use in generative settings as well.",
    body))

story.append(Paragraph("2 Related Work", h1))
story.append(Paragraph("2.1 Generative Adversarial Networks", h2))
story.append(Paragraph(
    "GANs consist of a generator and a discriminator trained in a minimax game. Numerous "
    "architectural improvements have been proposed to stabilize training and improve sample quality.",
    body))
story.append(Paragraph("2.2 Vision Transformers", h2))
story.append(Paragraph(
    "Vision Transformers (ViT) apply the standard transformer architecture directly to "
    "sequences of image patches, achieving strong results on image classification.",
    body))

story.append(Paragraph("3 ViTGAN", h1))
story.append(Paragraph("3.1 Discriminator", h2))
story.append(Paragraph(
    "The discriminator treats an image as a sequence of patches and processes them with a "
    "standard transformer encoder, followed by a classification head.", body))
story.append(Paragraph("3.2 Generator", h2))
story.append(Paragraph(
    "The generator maps a latent vector to a sequence of patch embeddings which are decoded "
    "into image patches and assembled into the final output image.", body))

story.append(Paragraph("4 Experiments", h1))
story.append(Paragraph(
    "We evaluate ViTGAN on CIFAR-10 and CelebA, reporting FID scores compared to convolutional "
    "baselines. Our model achieves competitive performance without any convolutional layers.",
    body))

story.append(Paragraph("5 Conclusion", h1))
story.append(Paragraph(
    "We presented ViTGAN, demonstrating that pure transformer architectures are viable for "
    "high-fidelity image generation, opening avenues for future convolution-free generative models.",
    body))

story.append(Paragraph("References", h1))
refs = [
    "[1] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention is all you need. NeurIPS, 2017.",
    "[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al. An image is worth 16x16 words: "
    "Transformers for image recognition at scale. ICLR, 2021.",
    "[3] I. Goodfellow, J. Pouget-Abadie, M. Mirza, et al. Generative adversarial networks. "
    "NeurIPS, 2014. doi:10.1145/3422622",
]
for r in refs:
    story.append(Paragraph(r, body))
    story.append(Spacer(1, 4))

doc.build(story)
print(f"wrote {out_path}")
