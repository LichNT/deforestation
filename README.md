# Deforestation Detection
This is a deforestation detection on RGB images using UNET.

## Training
    python train.py --num_epochs=100 --data_dir="forest/*.tif" --output_path=output
## Deploying
### Freeze the graph to a protobuf file
    python freeze_graph.py --input_graph=../output/graph.pb --input_checkpoint=../output/model.ckpt --output_node_names="Placeholder,div_1,Placeholder_2" --output_graph=../frozen_models/model2.pb  --input_binary=True
### Run inference on image(s)
    python infer.py --model=frozen_models/model.pb --input="./infer_images/0.png,./infer_images/1.png"
