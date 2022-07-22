# parts taken from the Lightning Quick Start tutorial and under their License.
from lit_email import LitEmail
from lit_torchdrift import LitTorchDrift
import lightning as L
import lightning_app.storage as storage
import model
import torchdrift
import pytorch_lightning as pl
import torch
import os
import time
import gradio
import glob
from lightning.app.components.serve import ServeGradio
from download import download_data
import torchvision.transforms as T
import torchvision

import lightning


class ImageServeGradio(ServeGradio):
    inputs = [
        gradio.inputs.Image(type="pil", shape=(20 * 28, 28), label="image input"),
    ]
    outputs = [
        gradio.outputs.Label(num_top_classes=10, label="first item in batch"),
        gradio.outputs.Textbox(label="prediction for all items"),
        gradio.outputs.Textbox(label="p-value"),
    ]

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.examples = None
        self.best_model_path = None
        self._transform = None
        self._labels = {idx: str(idx) for idx in range(10)}
        self._lit_email = LitEmail(
            email="name@email.com",
            password="yourPassword",
            smtp_server="beamnet.de",
        )

    def run(self):
        download_data(
            "https://pl-flash-data.s3.amazonaws.com/assets_lightning/images.tar.gz",
            "./",
        )
        imgs = [
            torchvision.io.read_image(f) / 255.0 for f in glob.glob("images/*.jpeg")
        ]
        imgs = torch.stack(imgs)

        all_same = imgs[torch.zeros(imgs.size(0), dtype=torch.int64)]

        imgs = imgs.permute(1, 2, 0, 3).reshape(28, -1)
        all_same = all_same.permute(1, 2, 0, 3).reshape(28, -1)
        imgs_c1 = torchdrift.data.functional.gaussian_blur(imgs[None, None], 1)[0, 0]
        imgs_c3 = torchdrift.data.functional.gaussian_blur(imgs[None, None], 3)[0, 0]
        imgs_c5 = torchdrift.data.functional.gaussian_blur(imgs[None, None], 5)[0, 0]
        img_names = [
            "images/imgs.png",
            "images/all_same.png",
            "images/imgs_c1.png",
            "images/imgs_c3.png",
            "images/imgs_c5.png",
        ]
        for n, i in zip(img_names, [imgs, all_same, imgs_c1, imgs_c3, imgs_c5]):
            torchvision.io.write_png((i[None] * 255).to(torch.uint8), n)
        self.examples = img_names
        self._transform = T.Normalize((0.1307,), (0.3081,))
        super().run()

    def predict(self, img):
        img = torchvision.transforms.functional.to_tensor(img)
        img = img.mean(0, keepdim=True)  # single channel...
        img = img.reshape(1, 28, -1, 28).permute(2, 0, 1, 3)
        img = self._transform(img)
        torch.save(img, "img.pt")
        prediction = torch.exp(self.model(img))
        self._lit_torchdrift.check(img)

        # Normally, we would not run the drift detector ourselves but let ListTorchDrift.check do its thing.
        # For UI/demonstration purposes, we run it a second time here...
        p_val = self._drift_detector(img).item()

        return [
            {self._labels[i]: prediction[0][i].item() for i in range(10)},
            ", ".join([self._labels[i] for i in prediction.argmax(1).tolist()]),
            f"p-value for drift: {p_val:.2f}",
        ]

    def build_model(self):
        detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
        reducer = torch.nn.Flatten()
        self._drift_detector = torch.nn.Sequential(
            reducer,
            detector,
        )
        drive = storage.Drive("lit://drive")
        drive.get("detector.pt")
        d = torch.load("detector.pt")
        detector.base_outputs = d["base_outputs"]
        detector.load_state_dict(d)

        self._lit_torchdrift = LitTorchDrift(
            detector=self._drift_detector,
            threshold=0.01,
            email_component=self._lit_email,
            alarm_email_address="alarm@example.com",
            alarm_for_low=True,  # low for p-value
        )

        m = model.ImageClassifier()
        drive.get("checkpoint.pt")
        m.load_state_dict(torch.load("checkpoint.pt"))
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        return m


class TrainModel(L.LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.drive = storage.Drive("lit://drive")
        self.done = False

    def run(self):
        m = model.ImageClassifier()
        datamodule = model.MNISTDataModule()
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(m, datamodule=datamodule)
        torch.save(m.state_dict(), "checkpoint.pt")
        self.drive.put("checkpoint.pt")
        os.remove("checkpoint.pt")
        self.done = True


class CalibrateDriftDetector(L.LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.drive = storage.Drive("lit://drive")
        self.done = False

    def run(self):
        self.drive.get("checkpoint.pt")
        m = model.ImageClassifier()
        m.eval()
        m.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
        detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
        data_module = model.MNISTDataModule()
        dl = data_module.train_dataloader(batch_size=20)
        torchdrift.utils.fit(
            dl, torch.nn.Flatten(), detector, num_batches=1, device="cpu"
        )
        torch.save(detector.state_dict(), "detector.pt")
        self.drive.put("detector.pt")
        os.remove("detector.pt")
        self.done = True


class LitApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.train_work = TrainModel()
        self.calib_work = CalibrateDriftDetector()
        self.serve_work = ImageServeGradio(lightning.CloudCompute("cpu"))
        self.last_print = -1

    def run(self):
        if not self.train_work.done:
            self.train_work.run()
        if self.train_work.done:
            self.calib_work.run()
        if self.calib_work.done:
            self.serve_work.run()

    def configure_layout(self):
        tab_1 = {"name": "Model training", "content": self.train_work}
        tab_2 = {"name": "Interactive demo", "content": self.serve_work}
        return [
            tab_2,
        ]  # tab_1]


app = L.LightningApp(LitApp())
