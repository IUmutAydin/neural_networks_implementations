import warnings
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from config import get_config, get_weights_file_path
from dataset import BilingualDataset, casual_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from model import Transformer, build_transformer


def greedy_decode(model: Transformer, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.int).to(device)

    while True:
        if decoder_input.size(1) > max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(0))

        decoder_output = model.decode(
            encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(decoder_output[:, -1])

        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.concat(decoder_input, torch.empty(
            1, 1).fill_(next_word.item()), dim=1).to(device)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, val_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    console_width = 80

    with torch.no_grad():
        for batch in val_ds:
            encoder_input = batch['encoder_input']
            encoder_mask = batch['encoder_mask']

            assert encoder_input.size(0) != 1

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        f'{config['datasource']}', f'{config['lang_src']}-{config['lang_tgt']}', split='train')

    tokenizer_src = get_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    valid_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, valid_ds_raw = random_split(
        ds_raw, [train_ds_size, valid_ds_size])

    train_dataset = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(valid_ds_raw, tokenizer_src, tokenizer_tgt,
                                   config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_src_len = 0
    max_tgt_len = 0

    for item in train_ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]).ids

        max_src_len = max(max_src_len, len(src_ids))
        max_tgt_len = max(max_tgt_len, len(tgt_ids))

    print(f'Max length of source sentence: {max_src_len}')
    print(f'Max length of target sentence: {max_tgt_len}')

    train_data_loader = DataLoader(train_dataset, config['batch_size'], True)
    valid_data_loader = DataLoader(val_dataset, 1, True)

    return train_data_loader, valid_data_loader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(
        src_vocab_size, tgt_vocab_size, config['seq_len'], config['d_model'])

    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_data_loader, valid_data_loader, tokenizer_src, tokenizer_tgt = get_ds(
        config)

    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    inital_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(inital_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(
            train_data_loader, f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(
                device)  # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(
                device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(
                device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(
                device)  # (batch, 1, seq_len, seq_len)

            # (batch, seq_len, d_model)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                # (batch, seq_len, d_model)
                encoder_output, encoder_mask, decoder_input, decoder_mask)
            # (batch, seq_len, vocab_size)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)  # (batch, seq_len)

            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar(
                tag='train loss', scalar_value=loss.item(), global_step=global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
