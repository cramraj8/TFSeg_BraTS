

def visualize_data():
    # https://stackoverflow.com/questions/46902190/subplots-within-subplots-two-8x8-subplots
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    pdf = PdfPages('./test.pdf')
    rows = 10
    cols = 1

    fig = plt.figure(figsize=(7, 8))
    gs0 = gridspec.GridSpec(1, 6)
    gs00 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[1])
    gs02 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[2])
    gs03 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[3])
    gs04 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[4])
    gs05 = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[5])

    import h5py
    h5f = h5py.File('results.h5', 'r')

    for i in range(rows * cols):

        ID = h5f['id_{}'.format(i)].value
        # print ID.value

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++  IMAGE  ++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # +++++++++++++++++++++++++++++  flair  ++++++++++++++++++++++++++++++++
        # ax00 = fig.add_subplot(gs00[i])
        ax00 = plt.subplot(gs00[i])
        img = h5f['image_{}'.format(i)][:]
        ax00.imshow(img[:, :, 0])
        ax00.set_aspect('auto')
        # ax00.set_xticks([])
        # ax00.set_yticks([])
        ax00.set_yticklabels([])
        ax00.set_xticklabels([])
        ax00.text(0.5, -0.15, "flair : {}".format(ID), size=7, ha="center",
                  transform=ax00.transAxes)

        # +++++++++++++++++++++++++++++  t1  ++++++++++++++++++++++++++++++++
        ax01 = plt.subplot(gs01[i])
        img = h5f['image_{}'.format(i)][:]
        ax01.imshow(img[:, :, 1])
        ax01.set_aspect('auto')
        ax01.set_yticklabels([])
        ax01.set_xticklabels([])
        ax01.text(0.5, -0.15, "t1 : {}".format(ID), size=7, ha="center",
                  transform=ax01.transAxes)

        # +++++++++++++++++++++++++++++  t1ce  ++++++++++++++++++++++++++++++++
        ax02 = plt.subplot(gs02[i])
        img = h5f['image_{}'.format(i)][:]
        ax02.imshow(img[:, :, 2])
        ax02.set_aspect('auto')
        ax02.set_yticklabels([])
        ax02.set_xticklabels([])
        ax02.text(0.5, -0.15, "t1ce : {}".format(ID), size=7, ha="center",
                  transform=ax02.transAxes)

        # +++++++++++++++++++++++++++++  t2  ++++++++++++++++++++++++++++++++
        ax03 = plt.subplot(gs03[i])
        img = h5f['image_{}'.format(i)][:]
        ax03.imshow(img[:, :, 3])
        ax03.set_aspect('auto')
        ax03.set_yticklabels([])
        ax03.set_xticklabels([])
        ax03.text(0.5, -0.15, "t2 : {}".format(ID), size=7, ha="center",
                  transform=ax03.transAxes)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++  LABEL  +++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ax01 = fig.add_subplot(gs01[i])
        ax04 = plt.subplot(gs04[i])
        lbl = h5f['label_{}'.format(i)][:]
        ax04.imshow(lbl)
        ax04.set_aspect('auto')
        # ax01.set_xticks([])
        # ax01.set_yticks([])
        ax04.set_yticklabels([])
        ax04.set_xticklabels([])
        ax04.text(0.5, -0.15, "label : {}".format(ID), size=7, ha="center",
                  transform=ax04.transAxes)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++  PREDICTION  ++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ax02 = fig.add_subplot(gs02[i])
        ax05 = plt.subplot(gs05[i])
        pred = h5f['pred_{}'.format(i)][:]
        ax05.imshow(pred)
        ax05.set_aspect('auto')
        # ax02.set_xticks([])
        # ax02.set_yticks([])
        ax05.set_yticklabels([])
        ax05.set_xticklabels([])
        ax05.text(0.5, -0.15, "predict : {}".format(ID), size=7, ha="center",
                  transform=ax05.transAxes)

    fig.tight_layout()
    plt.show()
    pdf.savefig(fig)
    pdf.close()


visualize_data()
