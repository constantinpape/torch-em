# source: https://www.synapse.org/#!Synapse:syn3193805/wiki/89480

# how to obtain the dataset?
#   step 1: join "Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge"
#       - link: https://www.synapse.org/#!Synapse:syn3193805
#   step 2: go to "Files" ->
#       - "Abdomen" -> "RawData.zip" to obtain all the abdominal CT scans
#       - "Cervix" -> "CervixRawData.zip" to obtain all the cervical CT scans
#   step 3: provide the path to the zipped file(s) to the respective datasets that takes care of it.


def unzip_inputs(zip_path):
    # unzip the inputs in the respective directories
    pass


def get_btcv_dataset(path):
    # implement the dataset from elf
    raise NotImplementedError


def get_btcv_loader(path):
    # get the dataset from elf, pass it to dataloader
    # make a split based on organs - by default uses ct scans from both organs
    # if specified, can use from either abdomen or cervix
    # NOTE: logic to resample the inputs
    # NOTE: logic for normalization for respective modalities
    #       - easiest: follow logic from nnunet (raw-trafo now in torch_em)
    raise NotImplementedError
