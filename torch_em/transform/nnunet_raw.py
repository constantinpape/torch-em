import json
from typing import Union
import numpy as np


class nnUNetRawTransform:
    """Apply nnUNet transformation to raw inptus.

    Adapted from nnUNetv2's `ImageNormalization`:
    https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/preprocessing/normalization

    You can use this class to apply the necessary raw transformations on input modalities.
    Currently supports transformations for CT and PET data. The inputs should be of dimension 2 * (H * W * D).
    - The first channel should be CT volume
    - The second channel should be PET volume

    Here's an example for how to use this class:
    ```python
    # Initialize the raw transform.
    raw_transform = nnUNetRawTransform(plans_file=".../nnUNetPlans.json")

    # Apply transformation on the inputs.
    patient_vol = np.concatenate(ct_vol, pet_vol)
    patient_transformed = raw_transform(patient_vol)
    ```

    Args:
        plans_file: The file with the nnUNet data plan.
        expected_dtype: The expected dtype of the input data.
        tolerance: The numerical tolerance.
        model_name: The name of the model.
    """
    def __init__(
        self,
        plans_file: str,
        expected_dtype: Union[np.dtype, str] = np.float32,
        tolerance: float = 1e-8,
        model_name: str = "3d_fullres",
    ):
        self.expected_dtype = expected_dtype
        self.tolerance = tolerance

        json_file = self.load_json(plans_file)
        self.intensity_properties = json_file["foreground_intensity_properties_per_channel"]
        self.per_channel_scheme = json_file["configurations"][model_name]["normalization_schemes"]

    def load_json(self, _file: str):
        """@private
        """
        # source: `batchgenerators.utilities.file_and_folder_operations`
        with open(_file, "r") as f:
            a = json.load(f)
        return a

    def ct_transform(self, channel, properties):
        """@private
        """
        mean = properties['mean']
        std = properties['std']
        lower_bound = properties['percentile_00_5']
        upper_bound = properties['percentile_99_5']

        transformed_channel = np.clip(channel, lower_bound, upper_bound)
        transformed_channel = (transformed_channel - mean) / max(std, self.tolerance)
        return transformed_channel

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """Returns the raw inputs after applying the pre-processing from nnUNet.

        Args:
            raw: The raw array inputs. Expectd a float array of shape M * (H * W * D), with M = number of modalities.

        Returns:
            The transformed raw inputs (the same shape as inputs).
        """
        assert raw.shape[0] == len(self.per_channel_scheme), "Number of channels & transforms from data plan must match"

        raw = raw.astype(self.expected_dtype)

        normalized_channels = []
        for idxx, (channel_transform, channel) in enumerate(zip(self.per_channel_scheme, raw)):
            properties = self.intensity_properties[str(idxx)]

            # get the correct transformation function, this can for example be a method of this class
            if channel_transform == "CTNormalization":
                channel = self.ct_transform(channel, properties)
            elif channel_transform in [
                "ZScoreNormalization", "NoNormalization", "RescaleTo01Normalization", "RGBTo01Normalization"
            ]:
                raise NotImplementedError(f"{channel_transform} is not supported by nnUNetRawTransform yet.")
            else:
                raise ValueError(f"Transform is not known: {channel_transform}.")

            normalized_channels.append(channel)

        return np.stack(normalized_channels)
