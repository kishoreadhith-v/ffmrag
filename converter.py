import torch
import timm
import torch.onnx

# Function to convert to ONNX
def convert_to_onnx(model, input_size=(1, 3, 336, 336), output_name="ImageClassifier.onnx"):
    model.eval()
    dummy_input = torch.randn(*input_size, requires_grad=True)
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        output_name,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,  # ONNX version for ViT compatibility
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['modelInput'],  # the model's input names
        output_names=['modelOutput'],  # the model's output names
        dynamic_axes={'modelInput': {0: 'batch_size'}, 'modelOutput': {0: 'batch_size'}}
    )
    print('Model has been converted to ONNX')

if __name__ == "__main__":
    # Load the ViT model architecture with correct number of classes
    model = timm.create_model('vit_large_patch14_clip_336', pretrained=False, num_classes=10000)
    # Load the state dict from the .pth file
    state_dict = torch.load('vit_large_patch14_clip_336.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    # Convert to ONNX
    convert_to_onnx(model)