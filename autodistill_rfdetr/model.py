from autodistill.detection import DetectionTargetModel
from rfdetr import RFDETRNano as NanoModel
from rfdetr import RFDETRSmall as SmallModel
from rfdetr import RFDETRMedium as MediumModel
from rfdetr import RFDETRLarge as LargeModel
from rfdetr import RFDETRSegPreview as SegPreviewModel
import supervision as sv

class RFDETR(DetectionTargetModel):
    def __init__(self):
        pass
        
    def predict(self, input:str, confidence=0.5) -> sv.Detections:
        self.model.optimize_for_inference()
        return self.model.predict(input, threshold=confidence)

    def train(self, dataset_yaml, epochs=25, output_dir="training-output", **kwargs):
        self.model.train(
            dataset_dir=dataset_yaml,
            epochs=epochs,
            batch_size=4,
            grad_accum_steps=4,
            lr=1e-4,
            output_dir=output_dir,
            **kwargs
        )

class RFDETRNano(RFDETR):
    def __init__(self, checkpoints = None):
        if checkpoints:
            self.model = NanoModel(checkpoints=checkpoints)
        else:
            self.model = NanoModel()

class RFDETRSmall(RFDETR):
    def __init__(self, checkpoints = None):
        if checkpoints:
            self.model = SmallModel(checkpoints=checkpoints)
        else:
            self.model = SmallModel()

class RFDETRMedium(RFDETR):
    def __init__(self, checkpoints = None):
        if checkpoints:
            self.model = MediumModel(checkpoints=checkpoints)
        else:
            self.model = MediumModel()

class RFDETRLarge(RFDETR):
    def __init__(self, checkpoints = None):
        if checkpoints:
            self.model = LargeModel(checkpoints=checkpoints)
        else:
            self.model = LargeModel()

class RFDETRSegPreview(RFDETR):
    def __init__(self, checkpoints = None):
        if checkpoints:
            self.model = SegPreviewModel(checkpoints=checkpoints)
        else:
            self.model = SegPreviewModel()