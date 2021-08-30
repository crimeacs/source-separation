import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, NullFormatter

import torch
import numpy as np
import io
import math
import tensorflow as tf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_tb_figure(signal, estimates):
    """Plots waveforms aquired as the result of source separation"""
    fig, ax = plt.subplots(nrows=estimates.shape[1] + 2, ncols=1, figsize=(12, 9))

    for i in range(0, signal.shape[1]):
        estimates_plot = estimates[0, i, :].squeeze().detach().cpu().numpy()
        signal_plot = signal[0, i, :].detach().cpu().numpy()

        # estimates_plot = normalize(estimates_plot[:,np.newaxis], axis=0).ravel()
        # signal_plot = normalize(signal_plot[:,np.newaxis], axis=0).ravel()

        ax[i].plot(estimates_plot, color='red')
        ax[i].plot(signal_plot, color='green', alpha=0.5)
        ax[i].set_title('Trace %i' % i)

    ax[i + 1].plot(np.sum(estimates[0, :, :].squeeze().detach().cpu().numpy(), axis=0), color='blue', alpha=0.5)
    ax[i + 1].plot(np.sum(signal[0, :, :].detach().cpu().numpy(), axis=0), color='green', alpha=0.5)
    ax[i + 1].set_title('Input signal - Mix of signals')

    residual = np.sum(signal[0, :, :].detach().cpu().numpy(), axis=0) ** 2 - np.sum(
        estimates[0, :, :].squeeze().detach().cpu().numpy(), axis=0) ** 2
    ax[i + 2].plot(residual, color='purple')
    ax[i + 2].set_title('Residual')

    plt.subplots_adjust(hspace=1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png');
    buf.seek(0);
    plt.close(fig)
    image = tf.image.decode_png(buf.getvalue(), channels=4);
    return torch.tensor(image.numpy())

def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)

    # frame  = signal.clone().detach().requires_grad_(True)
    frame = signal.new_tensor(frame, device=device).long()  # signal may be on GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)

    return result


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3:  # [B, C, T]
            results.append(input[:, :length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results


"""
Plotting routines for pysabeam.

.. module:: plotting

:author:
    Shahar Shani-Kadmiel (s.shanikadmiel@tudelft.nl)

:copyright:
    Shahar Shani-Kadmiel

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""


def add_colorbar_axes(position='right', size=0.1, pad=0.1, ax=None):
    """
    Add a colorbar axes relative to the active axes or ``ax``.
    """
    ax = ax or plt.gca()
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(position, size, pad)
    plt.sca(ax)

    return cax


def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2
    >>> _nearest_pow_2(15)
    16
    """
    a = np.power(2, np.ceil(np.log2(x)))
    b = np.power(2, np.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return int(a)
    else:
        return int(b)


time_fmt2type_map = {
    'relative': 'relative',
    'utc': 'matplotlib',
}


def plot_spectrogram(trace, time_fmt='relative', reftime=None,
                     xlim=None, ylim=None, vmin='0.005', vmax='0.2',
                     wtype='hann', wlen=None, overlap=0.9, ax=None,
                     scale='log', smooth=3, colorbar=True, celerities=False,
                     verbose=True, xlabel=False, **kwargs):
    """
    Convenience wrapper to :func:`~scipy.signal.spectrogram` combined
    with a plotting routine.

    Parameters
    ----------
    trace : :class:`~obspy.core.trace.Trace`
        An ObsPy Trace object.

    time_fmt : {'relative', 'utc'}
        Format of the time axis.

    reftime : None or UTCDateTime or str
        By default, waveforms are plotted relative to ``origintime`` if
        it exists as an attribute in the ``.stats``. If ``reftime`` is
        not ``None`` it is used as the reference time.
        If no ``origintime`` is found and no ``reftime`` is specified,
        the trace ``.stats.starttime`` attribute is used as the
        reference time.

    xlim, ylim : None or tuple
        If ``None``, data limits are used. Otherwise, a tuple
        ``(min, max)`` is expected.

    vmin, vmax : None or float, or str
        If ``None``, data limits are used. If a value is of type string,
        it is evaluated as ``float(value) * spect.max()``.

    wtype : str or tuple or array_like
        Desired window to use. If window is a string or tuple, it is
        passed to :func:`~scipy.signal.get_window` to generate the
        window values, which are DFT-even by default. See
        :func:`~scipy.signal.get_window` for a list of windows and
        required parameters. If window is array_like it will be used
        directly as the window and its length must be `wlen`. Defaults
        to a symmetric ``hann`` window.

    wlen : int
        Length of each segment in sample points. Defaults to
        sampling rate * 100. `wlen` is rounded to the nearest
        power of 2 whether specified or not.

    overlap: float
        Fraction of overlap between sliding windows. Defaults to 0.9
        which is 90%.

    ax : :class:`~matplotlib.axes.Axes`
        Axes to plot to. If ``None``, plotting is done to the active axes.

    scale : {'linear', 'log'}
        The axis scale type to apply.

    smooth : int
        Passed as size to :func:`~scipy.ndimage.filters.uniform_filter`.
        Set to ``False`` to forgo smoothing.

    colorbar : bool
        Plot the colorbar.

    celerities : sequence or bool
        Which celerities to plot above the time axis. If `True`, the
        default set is used.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Other parameters
    ----------------
    **kwargs : `~matplotlib.pyplot.imshow` and
        `~matplotlib.artist.Artist` properties.
    """
    # Compute spectrogram

    data = trace.data
    samp_rate = trace.stats.sampling_rate
    # catch origintime
    try:
        trace.stats.origintime
    except AttributeError:
        trace.stats.origintime = None
    reftime = reftime or trace.stats.origintime or trace.stats.starttime
    times = trace.times(time_fmt2type_map[time_fmt], reftime)

    wlen = wlen or samp_rate * 100.
    wlen = _nearest_pow_2(wlen)
    overlap = int(overlap * wlen)

    if verbose:
        print(('Computing spectrogram with {} samples long windows and '
               '{} points overlap.\n'
               'Total number of windows is {}...').format(
            wlen, overlap, int((data.size - wlen) / (wlen - overlap))))

    freq, time, spect = _spectrogram(data, samp_rate, wtype, wlen, overlap,
                                     scaling='spectrum', mode='magnitude')

    # plot it!
    ax = ax or plt.gca()
    plt.sca(ax)

    # calculate half bin width
    halfbin_time = 0.5 * (time[1] - time[0])
    halfbin_freq = 0.5 * (freq[1] - freq[0])

    spect = np.flipud(spect)
    if smooth:
        spect = uniform_filter(spect, smooth)

    xlim = xlim or (times[0], times[-1])
    reftime_ = times[0]

    ylim = ylim or (freq[0], 0.5 * freq[-1])

    # assign ``.time_fmt`` and ``.reftime`` attributes to the axes
    # object for later use
    ax.time_fmt = time_fmt
    ax.reftime = reftime

    t0 = reftime_ + time[0] - halfbin_time
    t1 = reftime_ + time[-1] + halfbin_time
    if xlabel == False:
        ax.set_xlabel('Time since {}, s'.format(reftime.strftime('%FT%T.%f')))
    else:
        ax.set_xlabel(xlabel)

    extent = (t0, t1, freq[0] - halfbin_freq, freq[-1] + halfbin_freq)

    kwargs['aspect'] = kwargs.get('aspect', 'auto')

    kwargs['vmax'] = vmax
    kwargs['vmin'] = vmin
    if isinstance(vmax, str):
        kwargs['vmax'] = float(vmax) * spect.max()
    if isinstance(vmin, str):
        kwargs['vmin'] = float(vmin) * spect.max()

    im = plt.imshow(spect, extent=extent, **kwargs)
    ax.set_xlim(xlim)
    ax.set_yscale(scale)
    ymin, ymax = ylim or (None, None)
    ymax = ymax or 0.5 * freq[-1]
    if scale == 'log':
        ymin = ymin or freq[1]
        ax.yaxis.set_major_locator(LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(LogLocator(subs='all', numticks=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ymin = ymin or freq[0]
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Frequency, Hz')
    ax.grid(True, axis='x', lw=0.5, color='k', alpha=0.1)

    if celerities is not False:
        try:
            if celerities:
                celerities = default_celerities
        except ValueError:
            pass
        plot_celerities(celerities, trace.stats.distance, ax)

    cb = None
    if colorbar:
        vmin, vmax = kwargs['vmin'], kwargs['vmax']
        extend = 'neither'
        if vmin > spect.min() and vmax < spect.max():
            extend = 'both'
        elif vmin > spect.min():
            extend = 'min'
        elif vmax < spect.max():
            extend = 'max'

        try:
            label = u'Power, {}²'.format(trace.stats.units)
        except AttributeError:
            label = u'Power, X²'
        cb = plt.colorbar(label=label, extend=extend,
                          aspect=10, pad=0.02)
        # cb = plt.colorbar(im, cax=add_colorbar_axes(),
        #                   label=u'Power, X²', extend=extend)
        cb.formatter.set_powerlimits((-1, 3))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

    plt.sca(ax)
    ax.cb = cb
    ax.keep_cb = True
    return im, cb


def process_snr(line):
    if type(line) != str:
        return [line, line, line]
    else:
        if ',' in line:
            x = ast.literal_eval(line)
            x = x[x != ''].astype(float)
            return x
        if ' ' in line:
            x = np.array(line[1:-1].split(' '))
            x = x[x != ''].astype(float)

            return x


def taper(waveform, func, max_percentage, max_length=None):
    waveform = waveform.squeeze()
    npts = len(waveform)

    # store all constraints for maximum taper length
    max_half_lenghts = []
    if max_percentage is not None:
        max_half_lenghts.append(int(max_percentage * npts))
    if max_length is not None:
        max_half_lenghts.append(int(max_length * self.stats.sampling_rate))

    # add full trace length to constraints
    max_half_lenghts.append(int(npts / 2))
    # select shortest acceptable window half-length
    wlen = min(max_half_lenghts)

    if 2 * wlen == npts:
        taper_sides = func(2 * wlen)
    else:
        taper_sides = func(2 * wlen + 1)
    taper = torch.tensor(np.hstack((taper_sides[:wlen], np.ones(npts - 2 * wlen),
                                    taper_sides[len(taper_sides) - wlen:])))

    waveform *= taper
    return waveform

def generate_data_samples(test_dataset):
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 8), sharey=False)
    colors = ["#03045e", "#023e8a",
              "#0077b6", "#0096c7",
              "#00b4d8", "#48cae4",
              "#90e0ef", "#ade8f4"]  # [::-2]

    count = 0
    for batch in test_dataset:
        sample, n_source = batch
        ax[count, 0].plot(sample[0], color=colors[count])
        ax[count, 0].set_title('Sample %i' % count);
        ax[count, 1].plot(sample[1], color=colors[count])
        ax[count, 1].set_title('Sample %i' % count);
        count += 1
        if count == 5:
            break

    plt.subplots_adjust(hspace=1)