from rich.progress import Progress
import argparse
import os
import numpy as np
import torch
import torchaudio


def is_audio_file(name):
    return name in ['.mp3', '.wav']


def save(data, path, **kwargs):
    s = data.cpu()
    if args.np:
        with open(path, 'w') as output:
            np.save(data, s.numpy())
    else:
        torch.save(data, path)


def process(f, idx, progress: Progress, task):
    progress.update(task, description=os.path.split(f)[0], total=0, completed=0)
    waveform, sr = torchaudio.load(f)
    # Take the first channel
    waveform = waveform[0, :].to(device)
    if sr != args.sr:
        waveform = torchaudio.transforms.Resample(sr, args.sr)(waveform)

    # Total length in milliseconds
    length = 1000 * waveform.size()[0] // args.sr

    mul_sr = args.sr // 1000
    offset = args.trim * mul_sr

    #                   <------------------Content-------------------->
    # Trimmed of start | Segment | Skipped | ... | Segment | (Skipped) | Trimmed of end
    seg_len = (args.segment_length + args.segment_spacing) * mul_sr
    n = (waveform.size()[0] - 2 * offset) // seg_len
    # An additional segment might fit without a subsequent skipped segment
    if waveform.size()[0] - 2 * offset - n * seg_len + args.segment_spacing * mul_sr > 0:
        n += 1

    progress.update(task, total=n, completed=0)
    for i in range(n):
        save(waveform[offset:args.segment_length * mul_sr], os.path.join(args.output_dir, f'{idx}-{i}.pt'))
        progress.update(task, completed=i)
        offset += seg_len


parser = argparse.ArgumentParser('Prepare a dataset from raw files')
parser.add_argument('-d', '--dir', default='./raw', help='Root directory for the files to be processed')
parser.add_argument('-od', '--output_dir', default='./dataset', help='Root directory the output')
parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                    help='Sample rate of the output, input will be resampled if it does not match')
parser.add_argument('-c', '--cpu', action='store_true',
                    help='Force CPU to be used, otherwise CUDA will be selected if available')
parser.add_argument('-l', '--segment_length', type=int, help='Length in ms of each extract')
parser.add_argument('-s', '--segment_spacing', type=int, help='Distance or offset after each extract in ms')
parser.add_argument('-t', '--trim', type=int, help='Ignore the first and last <value> ms of the file')
parser.add_argument('-np', action='store_true', help='Store as .npy (Numpy) instead of .pt (Pytorch)')

args = parser.parse_args()
device = 'cpu' if args.cpu or torch.cuda.is_available() else 'cuda'

if not os.path.isdir(args.dir):
    raise Exception(f'`{args.dir}` is not a directory')
if not os.path.isdir(args.output_dir):
    raise Exception(f'`{args.output_dir}` is not a directory')

files = list()
for (parent_path, _, filenames) in os.walk(args.dir):
    files += [os.path.join(parent_path, f) for f in filenames if is_audio_file(f)]

with Progress() as progress:
    overall_task = progress.add_task('Overall', total=len(files))
    current_file_task = progress.add_task('n/a', start=False)
    for idx, f in enumerate(files):
        process(f, idx, progress, current_file_task)
        progress.update(overall_task, advance=1)
