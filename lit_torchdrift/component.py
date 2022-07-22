import torchdrift
import torch
import logging
from typing import List
from lit_email import LitEmail

import lightning as L

logger = logging.getLogger(__name__)


class LitTorchDrift(L.LightningFlow):
    def __init__(
        self,
        detector: torchdrift.detectors.Detector,
        threshold: float,
        *,
        email_component: LitEmail,
        alarm_email_address: str,
        alarm_for_low: bool = True,
    ) -> None:

        super().__init__()
        self._detector = detector
        self.threshold = threshold
        self.alarm_for_low = alarm_for_low
        self.alarm_email_address = alarm_email_address
        self.email_component = email_component

    def check(self, features: torch.Tensor):
        self.run("check", features)

    def run(self, action, *args, **kwargs):
        if action == "check":
            self._check(*args, **kwargs)
        else:
            raise RuntimeError(f"unknown action {action}")

    def _check(self, features: torch.Tensor):
        value = self._detector(features).item()
        alarm = (
            (value < self.threshold) if self.alarm_for_low else (value > self.threshold)
        )
        logger.info(
            f"saw value {value}, with threshold {self.threshold}, the state is {alarm=}"
        )
        if alarm:
            try:
                self.email_component.send(
                    to_emails=self.alarm_email_address,
                    subject="Drift Alarm",
                    body=f"""Hello,

The drift value {value} exceeds the threshold {self.threshold}!

Best regards

TorchDrift
""",
                )
                logger.info("email alert sent")
            except Exception as e:
                logger.error("failed to send email...")
                logger.error(e)
