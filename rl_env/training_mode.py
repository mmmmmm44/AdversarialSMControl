from enum import Enum

class TrainingMode(Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"

    def __str__(self):
        return self.value