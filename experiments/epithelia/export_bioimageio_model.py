import z5py
from torch_em.util.modelzoo import export_bioimageio_model, get_default_citations, export_parser_helper


def _load_data():
    path = "/g/kreshuk/pape/Work/data/epethelia/test/per02_100.zarr"
    with z5py.File(path, "r") as f:
        raw = f["raw"][:]
    return raw


def export_to_bioimageio(checkpoint, output):
    input_data = _load_data()
    postprocessing = None
    offsets = [
        [-1, 0], [0, -1],
        [-3, 0], [0, -3],
        [-9, 0], [0, -9],
        [-27, 0], [0, -27]
    ]
    config = {"mws": {"offsets": offsets}}

    name = "EpitheliaAffinityModel"
    tags = ["u-net", "segmentation"]

    cite = get_default_citations(model="UNet2d", model_output="affinities")
    doc = "Affinity prediction for epithelia cells"

    export_bioimageio_model(
        checkpoint, output, input_data,
        name=name,
        authors=[{"name": "Constantin Pape; @constantinpape"}],
        tags=tags,
        license="CC-BY-4.0",
        documentation=doc,
        git_repo="https://github.com/constantinpape/torch-em.git",
        cite=cite,
        model_postprocessing=postprocessing,
        input_optional_parameters=False,
        config=config
    )


if __name__ == "__main__":
    parser = export_parser_helper()
    args = parser.parse_args()
    export_to_bioimageio(args.checkpoint, args.output)
