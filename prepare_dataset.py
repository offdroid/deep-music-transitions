from rich.progress import Progress
import argparse
import os
import numpy as np
import torch
import torchaudio


def is_audio_file(name):
    return name.lower().endswith('.mp3') or name.lower().endswith('.wav')


def save(data, path, **kwargs):
    if 'clone' in kwargs and kwargs['clone']:
        data = data.clone()
    s = data.cpu()
    if 'pt' in kwargs and kwargs['pt']:
        torch.save(s, path)
    else:
        np.save(path, s)


def process(f, idx, output_dir, segment_length, segment_spacing, trim=0, transform=None, one_file=False,
            progress: Progress = None, total=None, pt=True):
    if progress is not None:
        task = progress.add_task(f'{os.path.split(f)[-1]} ({idx + 1}/{total})',
                                 start=False)
    waveform, sr = torchaudio.load(f)
    # Take the first channel if more than one
    waveform = waveform[0, :].to(device)
    if sr != args.sample_rate:
        waveform = torchaudio.transforms.Resample(sr,
                                                  args.sample_rate)(waveform)

    # Convert ms values to absolute number of samples
    srps = args.sample_rate // 1000  # Sample rate per second
    segment_length *= srps
    segment_spacing *= srps
    if not isinstance(trim, tuple):
        trim = (abs(int(trim)) * srps, abs(int(trim)) * srps)
    else:
        trim = (abs(int(trim[0]) * srps), abs(int(trim[1])) * srps)

    #                   <------------------Content-------------------->
    # Trimmed of start | Segment | Skipped | ... | Segment | (Skipped) | Trimmed of end
    seg_len = segment_length + segment_spacing
    n = (waveform.size()[0] - trim[0] - trim[1]) // seg_len

    if progress is not None:
        progress.update(task, total=n, completed=0)
        progress.start_task(task)

    if one_file:
        if not trim[0] == trim[1] == 0:
            waveform = waveform[trim[0]:-trim[1]]
        waveform = waveform[:n * seg_len].view(-1, seg_len)[:, :segment_length]
        if transform is not None:
            waveform = transform(waveform)
        save(waveform, os.path.join(output_dir, f'{idx}.{"pt" if pt else "npy"}'), clone=True, pt=pt)
    else:
        offset = trim[0]
        for i in range(n):
            tmp = waveform[offset:segment_length * srps]
            if transform is not None:
                tmp = transform(tmp)
            save(tmp, os.path.join(output_dir, f'{idx}-{i}.{"pt" if pt else "npy"}'), pt=pt)
            offset += seg_len

            if progress is not None:
                progress.update(task, completed=i)

    if progress is not None:
        progress.update(task, completed=n)
        progress.stop_task(task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare a dataset from raw files')
    parser.add_argument('-d',
                        '--dir',
                        default='./raw',
                        help='Root directory for the files to be processed')
    parser.add_argument('-od',
                        '--output_dir',
                        default='./dataset',
                        help='Root directory the output')
    parser.add_argument(
        '-sr',
        '--sample_rate',
        type=int,
        default=16000,
        help=
        'Sample rate of the output, input will be resampled if it does not match'
    )
    parser.add_argument(
        '-c',
        '--cpu',
        action='store_true',
        help=
        'Force CPU to be used, otherwise CUDA will be selected, if available')
    parser.add_argument('-l',
                        '--segment_length',
                        default=2000,
                        type=int,
                        help='Length in ms of each extract')
    parser.add_argument('-s',
                        '--segment_spacing',
                        default=0,
                        type=int,
                        help='Distance or offset after each extract in ms')
    parser.add_argument(
        '-t',
        '--trim',
        default=0,
        type=int,
        help='Ignore the first and last <value> ms of the file')
    parser.add_argument('-pt',
                        default=False,
                        action='store_true',
                        help='Store as .pt (Pytorch) instead of .npy (Numpy)')
    parser.add_argument('--one_file',
                        default=False,
                        action='store_true',
                        help='Store all snippets of a file in one file')

    args = parser.parse_args()
    device = 'cpu' if args.cpu or torch.cuda.is_available() else 'cuda'

    if not os.path.isdir(args.dir):
        raise Exception(f'`{args.dir}` is not a directory')
    if not os.path.isdir(args.output_dir):
        raise Exception(f'`{args.output_dir}` is not a directory')

    # Process all files in args.dir or any of its subdirectories
    files = list()
    for (parent_path, _, filenames) in os.walk(args.dir):
        files += [
            os.path.join(parent_path, f) for f in filenames if is_audio_file(f)
        ]
    if len(files) == 0:
        print("No files found")
        exit(0)

    with Progress() as progress:
        for idx, f in enumerate(files):
            process(f, idx, args.output_dir, args.segment_length, args.segment_spacing, args.trim,
                    one_file=args.one_file, pt=args.pt, total=len(files), progress=progress)
