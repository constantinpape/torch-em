import json
import numpy as np


class nnUNetRawTransformBase:
    """nnUNetRawTransformBase is an interface to implement specific raw transforms for nnUNet.

    Adapted from: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/preprocessing/normalization
    """
    def __init__(
            self,
            plans_file: str,
            expected_dtype: type = np.float32,
            tolerance: float = 1e-8
    ):
        self.expected_dtype = expected_dtype
        self.tolerance = tolerance

        self.intensity_properties = self.load_json(plans_file)
        self.intensity_properties = self.intensity_properties["foreground_intensity_properties_per_channel"]

    def load_json(self, _file: str):
        # credits: `batchgenerators.utilities.file_and_folder_operations`
        with open(_file, 'r') as f:
            a = json.load(f)
        return a

    def __call__(
            self,
            raw: np.ndarray
    ) -> np.ndarray:  # the transformed raw inputs
        """Returns the raw inputs after applying the pre-processing from nnUNet.

        Args:
            raw: The raw array inputs
                Expectd a float array of shape M * (H * W * D) (where, M is the number of modalities)
        Returns:
            The transformed raw inputs (the same shape as inputs)
        """
        raise NotImplementedError("It's a class template for raw transforms from nnUNet. \
                                  Use a child class that implements the expected raw transform instead")


class nnUNetCTRawTransform(nnUNetRawTransformBase):
    """Apply transformation on the raw inputs for CT + PET channels (adapted from nnUNetv2's `CTNormalization`)

    You can use this class to apply the necessary raw transformations on CT and PET volume channels.
    Expectation: The inputs should be of dimension 2 * (H * W * D).
        - The first channel should be CT volume
        - The second channel should be PET volume

    Here's an example for how to use this class:
    ```python
    # Initialize the raw transform.
    raw_transform = nnUNetCTRawTransform(plans_file="...nnUNetplans.json")

    # Apply transformation on the inputs.
    patient_vol = np.concatenate(ct_vol, pet_vol)
    patient_transformed = raw_transform(patient_vol)
    ```
    """
    def __call__(
            self,
            raw: np.ndarray
    ) -> np.ndarray:
        assert raw.shape[0] == 2, "The current expectation is channels (modality) first. The fn currently supports for two modalities, namely CT and PET-CT (in the mentioned order)"

        assert self.intensity_properties is not None, \
            "Intensity properties are required here. Please make sure that you pass the `nnUNetplans.json correctly."

        raw = raw.astype(self.expected_dtype)

        transformed_raw = []
        # intensity properties for the respective modalities
        for idx in range(raw.shape[0]):
            props = self.intensity_properties[str(idx)]

            mean = props['mean']
            std = props['std']
            lower_bound = props['percentile_00_5']
            upper_bound = props['percentile_99_5']

            modality = np.clip(raw[idx, ...], lower_bound, upper_bound)
            modality = (modality - mean) / max(std, self.tolerance)
            transformed_raw.append(modality)

        transformed_raw = np.stack(transformed_raw)
        return transformed_raw
