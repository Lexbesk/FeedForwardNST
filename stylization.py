import os
import argparse
import torch
from models.style_transfer import StyleTransfer
import utils.utils as utils


def stylize(config):
    device = torch.device('mps')

    # load the model
    stylization_model = StyleTransfer().to(device)
    training_state = torch.load(os.path.join(config['model_binaries_path'], config['model_name']), map_location=device)
    state_dict = training_state['state_dict']
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    with torch.no_grad():
        if os.path.isdir(config['content_input']):
            pass

        else:
            img = utils.prepare_img_to_stylization(os.path.join(config['content_img_path'], config['content_input']),
                                                   config['width'], None, device)
            img = stylization_model(img).to("cpu")
            img = utils.post_process(img)
            utils.save(img, config['output_img_path'], "single_output")






if __name__ == "__main__":
    content_img_path = os.path.join(os.path.dirname(__file__), "data", "content-images")
    output_img_path = os.path.join(os.path.dirname(__file__), "data", "output-images")
    model_binaries_path = os.path.join(os.path.dirname(__file__), "models", "binaries")

    os.makedirs(output_img_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_input", type=str, help="Content image(s) to stylize", default='taj_mahal.jpg')
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--width", type=int, default=512)

    args = parser.parse_args()

    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)

    inference_config['content_img_path'] = content_img_path
    inference_config['output_img_path'] = output_img_path
    inference_config['model_binaries_path'] = model_binaries_path

    stylize(inference_config)

