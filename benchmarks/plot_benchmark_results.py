import collections
import itertools
import sys

import matplotlib as mpl
mpl.use('cairo')

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas


if sys.version_info[0] == 3:
    from functools import reduce
else:
    from __builtin__ import reduce

# Subtract the NoConv times from the rest of the model evaluation times
# to get the actual convolution-only times; then remove the column
def subtract_noconv(d):
    noconv = d['NoConv']
    for k in d.keys():
        # These are not actual measurements
        if k in ('Img', 'Krn', 'NoConv'):
            continue
        d[k] -= noconv
    d.drop('NoConv', 1, inplace=True)

def get_slope(x, y):
    r = np.corrcoef(x, y)[0][1]
    sx = np.std(x)
    sy = np.std(y)
    return r, r * sy / sx

def savefig(fname):
    fname = 'figs/' + fname
    print("Saving %s" % fname)
    plt.savefig(fname, bbox_inches='tight')

def _normalize_machine_name(machine):
    if machine == 'hyades':
        return 'pleiades'
    return machine

def _add_machine_name(ax, machine):
    twin = ax.twinx()
    twin.get_yaxis().set_ticks([])
    twin.set_ylabel(_normalize_machine_name(machine))

def plot_per_krn(data, ax, fname_pattern, **kwargs):

    xmin = data.Img.values.min()
    xmax = data.Img.values.max()

    for krn_size, krn_size_data in data.groupby('Krn'):

        krn_size_data = krn_size_data.set_index('Img').drop('Krn', 1)
        ax = krn_size_data.plot(xlim=(xmin, xmax), **kwargs)
        ax.set_ylabel('Times [s]')

        fname = fname_pattern % krn_size
        savefig(fname)

machine_order = ('sorrento', 'hyades', 'bolano')

def _plot_convolution_comparison_machine(mdata, column_data, gs_iter):

    idx = pandas.MultiIndex.from_tuples([(x, y) for x in mdata.Img.unique() for y in mdata.Krn.unique() if x >= y],
                                        names=('Image size', 'Krn'))

    m_data = pandas.DataFrame(mdata.values[:, 2:], index=idx, columns=mdata.columns[2:])
    superdata = pandas.DataFrame(index=idx)
    markers = list()
    colors = list()
    for new_name, (column, marker, color) in column_data.items():
        superdata[new_name] = m_data['BruteOld'] / m_data[column]
        markers.append(marker)
        colors.append(color)
    superdata = superdata.reorder_levels(['Krn', 'Image size'])

    lims = [superdata.values.min() / 1.1, superdata.values.max() * 1.1]
    for i, ksize in enumerate((25, 50, 100, 200)):
        ax = plt.subplot(next(gs_iter))
        ax.text(0.05, 0.90, 'ksize = %d' % (ksize,), transform=ax.transAxes,
                size='large')
        ax.set_ylim(lims)
        #superdata.loc[ksize].plot.bar(ax=ax, logy=True, sharex=True, legend=False)
        superdata.loc[ksize].plot(ax=ax, logy=True, sharex=True, legend=False)
        for line, marker, color in zip(ax.get_lines(), markers, colors):
            line.set_marker(marker)
            line.set_color(color)
        if i != 0:
            ax.get_yaxis().set_ticklabels([])
        else:
            ax.set_ylabel('Speedup')
        if i == 3:
            patches, labels = ax.get_legend_handles_labels()
            ax.legend(patches, labels, loc=4)
    return ax


def plot_convolution_comparison(data):

    names_per_machine = {
        'sorrento': {
            'Brute': ('Brute_16', 'o', '#1f77b4'),
            'OpenCL (float)': ('cl_00_f', 'v', '#ff7f0e'),
            'OpenCL (double)': ('cl_00_d', '^', '#2ca02c'),
            'FFT': ('FFT_1_16_Y', 'x', '#d62728')
        },
        'hyades': {
            'Brute': ('Brute_12', 'o', '#1f77b4'),
            'OpenCL (float)': ('cl_00_f', 'v', '#ff7f0e'),
            'OpenCL (double)': ('cl_00_d', '^', '#2ca02c'),
            'FFT': ('FFT_1_12_Y', 'x', '#d62728')
        },
        'bolano': {
            'Brute': ('Brute_2', 'o', '#1f77b4'),
            'OpenCL (float)': ('cl_00_f', 'v', '#ff7f0e'),
            'FFT': ('FFT_1_2_Y', 'x', '#d62728')
        },
    }

    # machine per machine
    figsize_base = 2
    for machine, names in names_per_machine.items():
        plt.figure(figsize=(figsize_base * 4, figsize_base))
        gs = gridspec.GridSpec(1, 4)
        gs.update(wspace=0, hspace=0) # set the spacing between axes
        _plot_convolution_comparison_machine(data[machine], names, iter(gs))
        savefig('conv_comparison_%s.png' % machine)

    # All in one!
    fig = plt.figure(figsize=(figsize_base * 4, figsize_base * 3))
    gs = gridspec.GridSpec(3, 4)
    gs.update(wspace=0, hspace=0) # set the spacing between axes
    gs_iter = iter(gs)
    for i, machine in enumerate(machine_order):
        last_col_ax = _plot_convolution_comparison_machine(data[machine], names_per_machine[machine], gs_iter)
        _add_machine_name(last_col_ax, machine)
        if i == 0:
            example_leg_handles_labels = last_col_ax.get_legend_handles_labels()
        last_col_ax.get_legend().set_visible(False)
    fig.legend(*example_leg_handles_labels,
               bbox_to_anchor=(0, 0, 1, 0),
               bbox_transform=fig.transFigure,
               loc=8, ncol=4, borderaxespad=-1., frameon=False)
    savefig('conv_comparison_summary.png')

def plot_opencl_comparison(data):

    figsize_base = 5

    # Show the comparison between convolvers created out of different OpenCL
    # devices and configurations
    common = data['hyades'].filter(regex='Krn|Img|BruteOld')
    nvidia = data['hyades'].filter(regex='cl_00_.*').rename(columns={'cl_00_f': 'NVidia (float)', 'cl_00_d': 'NVidia (double)'})
    intel_sdk = data['sorrento'].filter(regex='cl_00_.*').rename(columns={'cl_00_f': 'Intel SDK (float)', 'cl_00_d': 'Intel SDK (double)'})
    beignet = data['bolano'].filter(regex='cl_00_.*').rename(columns={'cl_00_f': 'Beignet'})
    pocl = data['bolano'].filter(regex='cl_10_.*').rename(columns={'cl_10_f': 'POCl (float)', 'cl_10_d': 'POCl (double)'})

    data = pandas.concat([common, nvidia, intel_sdk, beignet, pocl], axis=1)
    # calculate speed ups
    for col in ('NVidia (float)', 'NVidia (double)', 'Intel SDK (float)', 'Intel SDK (double)', 'Beignet', 'POCl (float)', 'POCl (double)'):
        data[col] = data['BruteOld'] / data[col]
    data.drop('BruteOld', 1, inplace=True)

    markers = ('.', ',', 'o', 'v', '^', 'P', 'x')
    ylims = (1, data.iloc[:, 2:].values.max() * 1.5)
    xlims = (data.Img.values.min(), data.Img.values.max())
    plt.figure(figsize=(figsize_base, figsize_base))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0, hspace=0) # set the spacing between axes
    for i, (krn_size, krn_size_data)  in enumerate(data.groupby('Krn')):

        ax = plt.subplot(gs[i])
        if i % 2 != 0:
            ax.yaxis.tick_right()
        else:
            ax.set_ylabel('Speedup')
        if i < 2:
            ax.xaxis.set_visible(False)

        krn_size_data = krn_size_data.set_index('Img').drop('Krn', 1)
        ax = krn_size_data.plot(ax=ax, logy=True)
        for line, marker in zip(ax.get_lines(), markers):
            line.set_marker(marker)
        if i == 0:
            ax.legend(ax.get_lines(), krn_size_data.columns)
        else:
            ax.legend().set_visible(False)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)

    savefig('ocl_platform_comparison.png')

def _plot_fft_reuse_comparison(ffts, ax=None, lims=None):

    per_reuse = ffts.reorder_levels(['Img', 'Krn', 'Threads', 'Effort', 'Reuse']).unstack()
    ax = per_reuse.plot.scatter(ax=ax, x='N', y='Y', zorder=10)
    if lims is None:
        lims = np.array([
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ])
    #ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.plot(lims, lims * 2/3, 'k-', alpha=0.75, zorder=0)

    # Calculate linear correlation, slope, and plot
    r, slope = get_slope(per_reuse['N'], per_reuse['Y'])
    ax.plot(lims, lims * slope, ':', alpha=0.75, zorder=0)
    ax.text(lims[1] / 1.1, lims[0] + 0.025, u'r = %.3f\ny = %.3fx' % (r, slope),
            verticalalignment='bottom', horizontalalignment='right')

    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return ax

def _plot_fft_effort_comparison(ffts, ax=None, lims=None):

    per_effort = ffts.reorder_levels(['Reuse', 'Krn', 'Img', 'Threads',
                                      'Effort']).unstack()
    per_effort_reuse_only = per_effort.loc['Y']

    # Plot per kernel. Start with krn = 25
    _data = per_effort_reuse_only.loc[25]
    ax = _data.plot(ax=ax, kind='scatter', x=0, y=1, zorder=10, color='DarkBlue',
                    label='25')
    _data = per_effort_reuse_only.loc[50]
    _data.plot(kind='scatter', x=0, y=1, zorder=10, color='DarkGreen', ax=ax,
               label='50')
    _data = per_effort_reuse_only.loc[100]
    _data.plot(kind='scatter', x=0, y=1, zorder=10, color='DarkOrange', ax=ax,
               label='100')
    _data = per_effort_reuse_only.loc[200]
    _data.plot(kind='scatter', x=0, y=1, zorder=10, color='DarkRed', ax=ax,
               label='200')
    ax.legend().set_visible(False)

    if lims is None:
        lims = np.array([
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ])

    # Draw 1:1 line
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    # Calculate linear correlation, slope, and plot
    r, slope = get_slope(per_effort_reuse_only.iloc[:, 0], per_effort_reuse_only.iloc[:, 1])
    ax.plot(lims, lims * slope, ':', alpha=0.75, zorder=0)
    ax.text(lims[1] / 1.1, lims[0] + 0.025, u'r = %.3f\ny = %.3fx' % (r, slope),
            verticalalignment='bottom', horizontalalignment='right')

    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return ax

def _plot_fft_thread_comparison(ffts, ax=None, *_):

    per_thread = ffts.reorder_levels(['Reuse', 'Effort', 'Img', 'Threads', 'Krn'])['Y']['1']
    colors = ('darkblue', 'darkgreen', 'darkorange', 'darkred', 'black', 'orange')
    markers = ('.', 'x', 'o', 'v', '^', 'P')
    imsizes = (100, 150, 200, 300, 400, 800)
    for imsize, color, marker in zip(imsizes, colors, markers):
        imdata = per_thread[imsize].unstack()
        imdata.index = imdata.index.astype(int)
        imdata = imdata.sort_index()
        ax = imdata.plot(ax=ax, legend=None, color=color, marker=marker,
                linewidth=0.8)
    ax.set_ylabel('Convolution time [s]')
    ax.set_xscale('log')
    handles = [mlines.Line2D([], [], color=color, label=str(imsize),
        marker=marker)
            for imsize, color, marker in zip(imsizes, colors, markers)]
    ax.legend(title='Image size', handles=list(reversed(handles)), fontsize=8)
    return ax

def _plot_fft_comparison_per_machine_plus_summary(ffts, f, xlabel, ylabel, fname_base):

    figsize_base = 2.5
    max_lims = [0, 0]
    for machine, _ffts in ffts.items():
        ax = f(_ffts)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        savefig(fname_base + '_%s.png' % machine)
        max_lims = np.array([
            np.min([ax.get_xlim(), max_lims]),
            np.max([ax.get_xlim(), max_lims])
        ])

    plt.figure(figsize=(len(ffts) * figsize_base, figsize_base))
    gs = gridspec.GridSpec(1, len(ffts))
    gs.update(wspace=0, hspace=0) # set the spacing between axes
    for i, (cell, (machine, _ffts)) in enumerate(zip(gs, ffts.items())):
        ax = plt.subplot(cell)
        ax = f(_ffts, ax, max_lims)
        ax.set_xlabel(xlabel if i == 1 else '')
        ax.set_ylabel(ylabel if i == 0 else '')
        if i != 0:
            ax.get_yaxis().set_ticklabels([])
    savefig(fname_base + '_summary.png')

    plt.figure(figsize=(figsize_base, len(ffts) * figsize_base))
    gs = gridspec.GridSpec(len(ffts), 1)
    gs.update(wspace=0, hspace=0) # set the spacing between axes
    for i, (cell, (machine, _ffts)) in enumerate(zip(gs, ffts.items())):
        ax = plt.subplot(cell)
        ax = f(_ffts, ax, max_lims)
        ax.set_xlabel(xlabel if i == 0 else '')
        ax.set_ylabel(ylabel)
        if i != len(ffts.items()) - 1:
            ax.get_xaxis().set_ticklabels([])
    savefig(fname_base + '_summary_vertical.png')

def plot_fft_comparison(data):

    # Generic cleanup
    ffts = collections.OrderedDict()
    for machine in machine_order:
        all_data = data[machine]

        # Consider nothing else than FFT convolvers
        to_remove = [k for k in all_data.keys() if not k.startswith('FFT_') and k not in ('Krn', 'Img')]
        _ffts = all_data.drop(to_remove, 1)

        # Index by Img, Krn, Effort, Thread and reusage
        idx = pandas.MultiIndex.from_tuples([(x, y) for x in _ffts.Img.unique() for y in _ffts.Krn.unique() if x >= y],
                                            names=('Img', 'Krn'))
        cols = _ffts.columns[2:].str.extract(r'FFT_(.+)_(\d+)_(Y|N)', expand=True)
        columns = pandas.MultiIndex.from_arrays((cols[0], cols[1], cols[2]),
                                                names=('Effort', 'Threads', 'Reuse'))
        _ffts = pandas.DataFrame(_ffts.iloc[:, 2:].values, index=idx, columns=columns)
        _ffts = _ffts.stack().stack().stack()
        ffts[machine] = _ffts

    _plot_fft_comparison_per_machine_plus_summary(ffts, _plot_fft_reuse_comparison, 'No reuse time [s]', 'Reuse time [s]', 'fftconv_reuse_comparison')
    _plot_fft_comparison_per_machine_plus_summary(ffts, _plot_fft_effort_comparison, 'Effort = ESTIMATE, time [s]', 'Effort = MEASURE, time [s]', 'fftconv_effort_comparison')
    _plot_fft_comparison_per_machine_plus_summary(ffts, _plot_fft_thread_comparison, '# Threads', 'Convolution time [s]', 'fftconv_threadcount_comparison')


def _old_plot_ptimes(machine, cols, grid, figsize, imsize):
    na_values = ['%8s' % ('[E%d]' % n) for n in range(100)]
    ptimes = pandas.read_fwf("results/%s-profiles-%d.csv" % (machine, imsize), comment='#', na_values=na_values)

    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(*grid)
    #gs.update(wspace=0, hspace=0) # set the spacing between axes
    for i, (col, label) in enumerate(cols.items()):
        series = ptimes['CPU'] / ptimes[col]
        ax = plt.subplot(gs[i])
        ax = series.hist(bins=100, ax=ax, log=True)
        ax.set_title(label)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        stats = 'median = %.2f\navg = %.2f\nstd = %.2f' % (series.median(), series.mean(), series.std())
        ax.text(xlims[1] / 1.1, ylims[1] / 1.1, stats,
                verticalalignment='top', horizontalalignment='right')

    savefig('profile_speedup_%s_%d' % (machine, imsize))

def _load_ptimes(machine, imsize, cols, labels):
    na_values = ['%8s' % ('[E%d]' % n) for n in range(100)]
    ptimes = pandas.read_fwf("results/%s-profiles-%d.csv" % (machine, imsize), comment='#', na_values=na_values)
    ptimes = ptimes.set_index(['nser', 'ang', 'axrat', 're', 'box'])
    for col, label in zip(cols, labels):
        ptimes[label] = ptimes['CPU'] / ptimes[col]
    return ptimes

def _plot_ptimes(ptimes, cols, labels, ax=None):
    ax = ptimes.boxplot(ax=ax, column=list(reversed(labels)), vert=False,
            whis=[1, 99], flierprops={'markersize': 0.5})
    ax.set_xscale('log')
    ax.set_xlabel('Speedup')
    return ax

def plot_ptimes():

    im_sizes = 200, 400, 800
    desc = {
        'sorrento': (
            ['CL_00_f', 'CL_00_d', 'OMP_2', 'OMP_4', 'OMP_8', 'OMP_16'],
            ['Intel SDK (float)', 'Intel SDK (double)', 'OpenMP (2 threads)', 'OpenMP (4 threads)', 'OpenMP (8 threads)', 'OpenMP (16 threads)']
        ),
        'bolano': (
            ['CL_00_f', 'CL_10_f', 'CL_10_d', 'OMP_2'],
            ['Beignet', 'POCl (float)', 'POCl (double)', 'OpenMP (2 threads)']
        ),
        'hyades': (
            ['CL_00_f', 'CL_00_d', 'OMP_2', 'OMP_4', 'OMP_8', 'OMP_12'],
            ['NVidia (float)', 'NVidia (double)', 'OpenMP (2 threads)', 'OpenMP (4 threads)', 'OpenMP (8 threads)', 'OpenMP (12 threads)']
        )
    }

    ptimes = {(machine, imsize): _load_ptimes(machine, imsize, cols, labels)
              for (machine, (cols, labels)), imsize in itertools.product(desc.items(), im_sizes)}

    # Plot individually
    for (m, (cols, labels)), im_size in itertools.product(desc.items(), im_sizes):
        plt.figure()
        _plot_ptimes(ptimes[(m, im_size)], cols, labels)
        savefig('profile_speedup_%s_%d.png' % (m, im_size))

    # Now all combined into a single 3x3 plot!
    x0 = reduce(min, map(lambda x: x.values.min(), ptimes.values())) / 1.1
    x1 = reduce(max, map(lambda x: x.values.max(), ptimes.values())) * 1.1
    x0 = 0.04
    plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0, hspace=0) # set the spacing between axes
    for i, ((m, im_size), _gs) in enumerate(zip(itertools.product(machine_order, im_sizes), gs)):
        cols, labels = desc[m]
        ax = plt.subplot(_gs)
        ax = _plot_ptimes(ptimes[(m, im_size)], cols, labels, ax=ax)
        ax.set_xlim((x0, x1))
        if (i % 3) != 0:
            ax.get_yaxis().set_ticklabels([])
        if (i % 3) == 2:
            _add_machine_name(ax, m)
        if i / 3 < 2:
            ax.get_xaxis().set_ticklabels([])
    savefig('profile_speedup_summary.png')

def _plot_bruteforce_speedups(x, plot_avx, ax=None):
    sw = x.loc['brute-old'] / x.loc['no_simd']
    ax = sw.plot(ax=ax)
    sse2 = x.loc['brute-old'] / x.loc['generic']
    ax = sse2.plot(ax=ax)
    if plot_avx:
        avx = x.loc['brute-old'] / x.loc['native']
        ax = avx.plot(ax=ax)
    ax.set_ylabel('Speedup')
    return ax

def plot_bruteforce_comparison():
    # Load data
    bf_speedups = {}
    for machine in machine_order:
        data = pandas.read_csv('results/brute-force-%s.csv' % machine)
        idx = pandas.MultiIndex.from_tuples([(t, i, p)
                                             for t in data.type.unique()
                                             for i in data.imsize.unique()
                                             for p in data.psf.unique() if i >= p],
                                             names=('Type', 'Img', 'Krn'))
        columns = pandas.MultiIndex.from_arrays((data.columns[-1:],))
        bf_speedups[machine] = pandas.DataFrame(data.iloc[:,-1:].values, index=idx, columns=columns)

    # Single plot for all of them
    markers = ('.', 'x', '^')
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0, hspace=0)
    for i, (machine, cell) in enumerate(zip(machine_order, gs)):
        ax = plt.subplot(cell)
        ax = _plot_bruteforce_speedups(bf_speedups[machine], machine != 'hyades', ax=ax)
        for line, marker in zip(ax.get_lines(), markers):
            line.set_marker(marker)
        if i != 0:
            ax.legend().remove()
        else:
            ax.legend(['C++', 'SSE2', 'AVX'])
        _add_machine_name(ax, machine)
    savefig('brute_force_convolution_comparison_summary.png')

if __name__ == '__main__':

    na_values = ['%8s' % ('[E%d]' % n) for n in range(100)]
    data = {}
    for f in ('bolano', 'hyades', 'sorrento'):
        data[f] = pandas.read_fwf('results/%s-convolution.csv' % f, comment='#', na_values=na_values)

    # Some general cleanups
    for d in data.values():

        # Column names should be trimmed
        d.rename(columns=lambda x: x.strip(), inplace=True)

        # Don't consider the Local OpenCL kernels
        to_remove = [k for k in d.keys() if k.startswith('Lcl_')]
        d.drop(to_remove, 1, inplace=True)

        # and subtract the no-convolution times
        subtract_noconv(d)

    plot_bruteforce_comparison()
    plot_fft_comparison(data)
    plot_convolution_comparison(data)
    plot_opencl_comparison(data, None)
    plot_ptimes()
