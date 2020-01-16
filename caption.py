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
	


