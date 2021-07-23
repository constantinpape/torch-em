mkdir -p exported_models_mws

echo "Export isbi2012 model"
cd neuron-segmentation/isbi2012
python export_bioimageio_model.py -c checkpoints/affinity-model -i /g/kreshuk/data/isbi2012_challenge/isbi2012_test_volume.h5 -o ../../exported_models_mws/isbi2012 -a 0 -f torchscript onnx
cd ../..
