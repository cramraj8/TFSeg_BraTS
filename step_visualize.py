

def visualize_data():
    # https://stackoverflow.com/questions/46902190/subplots-within-subplots-two-8x8-subplots
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    pdf = PdfPages('./test.pdf')
    rows = 10
    cols = 1

    fig = plt.figure(figsize=(6, 6))
    gs0 = gridspec.GridSpec(1, 3)
    gs00 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[1])
    gs02 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[2])

    import h5py
    h5f = h5py.File('results.h5', 'r')

    for i in range(rows * cols):

        ID = h5f['id_{}'.format(i)].value
        # print ID.value

        # ax00 = fig.add_subplot(gs00[i])
        ax00 = plt.subplot(gs00[i])
        img = h5f['image_{}'.format(i)][:]
        ax00.imshow(img)
        ax00.set_aspect('auto')
        # ax00.set_xticks([])
        # ax00.set_yticks([])
        ax00.set_yticklabels([])
        ax00.set_xticklabels([])
        ax00.text(0.5, -0.15, "image : {}".format(ID), size=8, ha="center",
                  transform=ax00.transAxes)

        # ax01 = fig.add_subplot(gs01[i])
        ax01 = plt.subplot(gs01[i])
        lbl = h5f['label_{}'.format(i)][:]
        ax01.imshow(lbl)
        ax01.set_aspect('auto')
        # ax01.set_xticks([])
        # ax01.set_yticks([])
        ax01.set_yticklabels([])
        ax01.set_xticklabels([])
        ax01.text(0.5, -0.15, "label : {}".format(ID), size=8, ha="center",
                  transform=ax01.transAxes)

        # ax02 = fig.add_subplot(gs02[i])
        ax02 = plt.subplot(gs02[i])
        pred = h5f['pred_{}'.format(i)][:]
        ax02.imshow(pred)
        ax02.set_aspect('auto')
        # ax02.set_xticks([])
        # ax02.set_yticks([])
        ax02.set_yticklabels([])
        ax02.set_xticklabels([])
        ax02.text(0.5, -0.15, "predict : {}".format(ID), size=8, ha="center",
                  transform=ax02.transAxes)

    fig.tight_layout()
    plt.show()
    pdf.savefig(fig)
    pdf.close()


visualize_data()
