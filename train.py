import os
import argparse
import utils.utils as utils
import torch
from models.style_transfer import StyleTransfer
from models.transform import TransformerNet
from models.vgg import Vgg16
from torch.utils.tensorboard import SummaryWriter
import time


def train(config):
    writer = SummaryWriter()
    device = torch.device("cuda")
    data_loader = utils.get_data_loader(config)
    # stylization_model = StyleTransfer().train().to(device)
    stylization_model = TransformerNet().train().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    optimizer = torch.optim.Adam(stylization_model.parameters())

    style_img = utils.prepare_img_to_stylization(os.path.join(config['style_images_path'], config['style_img_name']),
                                                 None, None, device)

    target_style_feature_maps = vgg(style_img)
    target_style_grams = [utils.gram_matrix(i) for i in target_style_feature_maps]
    acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]
    ts = time.time()

    for epoch in range(config['num_of_epochs']):
        for j, (batch, _) in enumerate(data_loader):
            batch = batch.to(device)
            stylized_batch = stylization_model(batch)

            batch_feature_maps = vgg(batch)
            style_feature_maps = vgg(stylized_batch)

            # content_loss
            content_feature_target = batch_feature_maps.relu2_2
            content_feature = style_feature_maps.relu2_2
            content_loss = torch.nn.MSELoss(reduction='mean')(content_feature_target, content_feature) * config['content_weight']

            # style_loss
            style_grams = [utils.gram_matrix(i) for i in style_feature_maps]
            # print(style_grams[0][0, 0])
            style_loss = 0.0
            for k in range(len(style_grams)):
                # print(style_grams[k].shape, target_style_grams[k].shape)
                style_loss += torch.nn.MSELoss(reduction='mean')(style_grams[k], target_style_grams[k])
                # print(style_loss)
            style_loss = style_loss * config['style_weight'] / len(style_grams)

            # tv_loss
            tv_loss = config['tv_weight'] * utils.total_variation(stylized_batch)

            loss = content_loss + style_loss + tv_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()
            acc_tv_loss += tv_loss.item()

            if config['enable_tensorboard']:
                writer.add_scalar('Loss/content_loss', content_loss.item(), len(data_loader) * epoch + j + 1)
                writer.add_scalar('Loss/style_loss', style_loss.item(), len(data_loader) * epoch + j + 1)
                writer.add_scalar('Loss/tv_loss', content_loss.item(), len(data_loader) * epoch + j + 1)
                if j % config['image_log_freq'] == 0:
                    stylized = utils.post_process(stylized_batch.detach().to('cpu'))
                    writer.add_image('stylized_img', stylized[:, :, ::-1], len(data_loader) * epoch + j + 1, dataformats='HWC')
            if config['console_log_freq'] is not None and j % config[
                    'console_log_freq'] == 0:
                print(
                        f'time elapsed={(time.time() - ts) / 60:.2f}[min]|epoch={epoch + 1}|batch=[{j + 1}/{len(data_loader)}]|c-loss={acc_content_loss / config["console_log_freq"]}|s-loss={acc_style_loss / config["console_log_freq"]}|tv-loss={acc_tv_loss / config["console_log_freq"]}|total loss={(acc_content_loss + acc_style_loss + acc_tv_loss) / config["console_log_freq"]}')
                acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]

    training_state = utils.get_training_metadata(config)
    training_state["state_dict"] = stylization_model.state_dict()
    training_state["optimizer_state"] = optimizer.state_dict()
    model_name = f"style_{config['style_img_name'].split('.')[0]}_cw_{str(config['content_weight'])}_sw_{str(config['style_weight'])}_tw_{str(config['tv_weight'])}.pth"
    torch.save(training_state, os.path.join(config['model_binaries_path'], model_name))


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
    style_images_path = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    image_size = 256  # training images from MS COCO are resized to image_size x image_size
    batch_size = 4

    assert os.path.exists(dataset_path), f'MS COCO missing. Download the dataset using resource_downloader.py script.'
    os.makedirs(model_binaries_path, exist_ok=True)

    #
    # Modifiable args - feel free to play with these (only a small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    # training related
    parser.add_argument("--style_img_name", type=str, help="style image name that will be used for training", default='edtaonisl.jpg')
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e0)  # you don't need to change this one just play with style loss
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=4e5)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=0)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs ", default=2)
    parser.add_argument("--subset_size", type=int, help="number of MS COCO images (NOT BATCHES) to use, default is all (~83k)(specified by None)", default=None)
    # logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging (scalars + images)", default=True)
    parser.add_argument("--image_log_freq", type=int, help="tensorboard image logging (batch) frequency - enable_tensorboard must be True to use", default=100)
    parser.add_argument("--console_log_freq", type=int, help="logging to output console (batch) frequency", default=500)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (batch) frequency", default=2000)
    args = parser.parse_args()

    checkpoints_path = os.path.join(checkpoints_root_path, args.style_img_name.split('.')[0])
    if args.checkpoint_freq is not None:
        os.makedirs(checkpoints_path, exist_ok=True)

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['dataset_path'] = dataset_path
    training_config['style_images_path'] = style_images_path
    training_config['model_binaries_path'] = model_binaries_path
    training_config['checkpoints_path'] = checkpoints_path
    training_config['image_size'] = image_size
    training_config['batch_size'] = batch_size

    # Original J.Johnson's training with improved transformer net architecture
    train(training_config)