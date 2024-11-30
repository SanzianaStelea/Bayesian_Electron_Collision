def cornerplot(chain, bins, colors, file_path, labels=None, levels, quantiles):
    """
    Function to plot a corner plot of the chain.
    :param chain: The chain to plot.
    :param bins: Number of bins to use in the histograms.
    :param colors: Colors to use in the plot.
    :param file_path: Path to save the plot.
    :param labels: Labels for the parameters.
    :param levels: Levels for the contours.
    :param quantiles: Quantiles to plot.
    :param fig: Figure to plot on.
    :return: None.
    """
    fig = plt.figure()
    fig = corner.corner(
        chain,
        bins=bins,
        labels=labels,
        color=colors,
        levels=levels, # Credible contours corresponding
                                                # to 1 and 2 sigma in 2D
        quantiles=quantiles,
        fig=fig
    )
    fig.subplots_adjust(hspace=0.25)
    fig.get_axes()[0].plot([], [], c="C0", label="Samples from the posterior")
    fig.get_axes()[0].plot([], [], c="C1", label="True parameters")
    fig.get_axes()[0].legend(loc=2, bbox_to_anchor=(1, 1))

    fig.tight_layout(pad=2.0)

    for ax in fig.get_axes():
        ax.xaxis.label.set_size(12)  # Optional: Adjust font size if needed
        ax.yaxis.label.set_size(12)
        ax.xaxis.labelpad = 80       # Adjust label padding for x-axis
        ax.yaxis.labelpad = 30       # Adjust label padding for y-axis

    # for ax in fig.get_axes():
    #     ax.tick_params(axis="both", which="both", length=0)  # Remove tick marks
    #     ax.set_xticklabels([])  # Remove x-axis tick labels
    #     ax.set_yticklabels([])  # Remove y-axis tick labels

    fig.set_size_inches(14, 9)
    fig.savefig(file_path + '20241130_corner_plot_gauss_poly_bg.png')