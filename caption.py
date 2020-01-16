import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

data_folder = 'final_dataset'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file ='model_poids'#'checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'#'BEST_9checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'#'checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'# 'BEST_34checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint

word_map_file = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
emb_dim = 1024  # dimension of word embeddings
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhe

# Load model
# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

torch.nn.Module.dump_patches = True
decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
decoder.load_state_dict(torch.load(checkpoint_file))
decoder = decoder.to(device)
decoder.eval()
print(type(decoder))


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch,data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)


    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
        # One epoch's training
    caption(train_loader=train_loader,decoder=decoder,criterion_ce = criterion_ce,criterion_dis=criterion_dis,decoder_optimizer=decoder_optimizer,epoch=epoch)


def caption(train_loader, decoder, criterion_ce, criterion_dis, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
	print(type(imgs))
	print(imgs)
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores, scores_d,caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
        
        #Max-pooling across predicted words across time steps for discriminative supervision
        scores_d = scores_d.max(1)[0]

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        targets_d = torch.zeros(scores_d.size(0),scores_d.size(1)).to(device)
        targets_d.fill_(-1)

        for length in decode_lengths:
            targets_d[:,:length-1] = targets[:,:length-1]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores= pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
	
	print(scores)
	print(targets)
	

	

	
if __name__ == '__main__':
    main()
