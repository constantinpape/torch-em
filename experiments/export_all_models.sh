mkdir -p exported_models

echo "Export covid-if model"

echo "Export cremi model"

echo "Export dsb model"
cd dsb
python export_bioimageio_model.py -c checkpoints/dsb-affinity-model -i /scratch/pape/dsb/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif -o ../exported_models/dsb_boundaries -a 1 -f torchscript
cd ..

echo "Export isbi2012 model"

echo "Export mito-em model"

echo "Export plantseg models"
cd plantseg
# python export_bioimageio_model.py -c ovules/checkpoints/affinity_model2d -i /g/kreshuk/wolny/Datasets/Ovules/GT2x/val/N_420_ds2x.h5 -o ../exported_models/ovules_boundaries -a 1 -f torchscript
cd ..

echo "Export platynereis models"
