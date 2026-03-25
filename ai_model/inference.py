import torch
from PIL import Image
from torchvision import transforms
from models.generator import Generator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Generator().to(device)
model.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def generate(img_path):
    print("👉 Loading image from:", img_path)

    if not os.path.exists(img_path):
        print("❌ ERROR: Image path not found!")
        return

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)

    out = (out.squeeze().cpu() * 0.5 + 0.5)

    output_path = "result.jpg"
    transforms.ToPILImage()(out).save(output_path)

    print("✅ Output saved at:", output_path)


if __name__ == "__main__":
    generate(r"D:\AI_Projects\AI_Room_Renovator\Dataset\ROOM_DATASET\train\input\bath_2.jpg")
