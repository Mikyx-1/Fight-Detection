from dataset import create_dataset
from model import build_model
import pathlib

EPOCHS = 50



model = build_model()

subset_paths = {"train": pathlib.Path("C:/Users/DELL Inspiron/Downloads/UCF101_videos/train"),
                "validation": pathlib.Path("C:/Users/DELL Inspiron/Downloads/UCF101_videos/validation")}  

train_ds = create_dataset(subset_paths["train"])
val_ds = create_dataset(subset_paths["validation"])

model.fit(train_ds, epochs=EPOCHS, validation_data = val_ds)
