

import numpy as np
import random
from glob import glob
from skimage import io
import os

import config
import h5py


np.random.seed(5)  # for reproducibility


class BrainPipeline(object):
    '''
    A class for processing brain scans for one patient
    INPUT:  (1) filepath 'path': path to directory of one patient. Contains following mha files:
            flair, t1, t1c, t2, ground truth (gt)
            (2) bool 'n4itk': True to use n4itk normed t1 scans (defaults to True)
            (3) bool 'n4itk_apply': True to apply and save n4itk filter to t1 and t1c scans for given patient. This will only work if the
    '''

    def __init__(self, path, n4itk=True, n4itk_apply=False):
        self.path = path
        # self.n4itk = n4itk
        # self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        _, self.slices_by_slice = self.read_scans()
        self.normed_slices = self.norm_slices()
        print(self.slices_by_slice.shape)
        print(self.normed_slices.shape)

    def read_scans(self):
        '''
        goes into each modality in patient directory and loads individual scans.
        transforms scans of same slice into strip of 5 images
        '''
        print 'Loading scans...'
        slices_by_mode = np.zeros((5, 155, 240, 240))
        slices_by_slice = np.zeros((155, 5, 240, 240))

        flair = glob(self.path + '*_flair.nii.gz')
        print flair
        t2 = glob(self.path + '/*_t2.nii.gz')
        print t2
        gt = glob(self.path + '/*_seg.nii.gz')
        print gt
        t1c = glob(self.path + '/*_t1ce.nii.gz')
        print t1c
        t1 = glob(self.path + '/*_t1.nii.gz')

        scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
        # if self.n4itk_apply:
        #     print '-> Applyling bias correction...'
        #     for t1_path in t1:
        #         self.n4itk_norm(t1_path)  # normalize files
        #     scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
        # elif self.n4itk:
        #     scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]

        for scan_idx in xrange(5):
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
        for mode_ix in xrange(slices_by_mode.shape[0]):         # modes 1 thru 5
            for slice_ix in xrange(slices_by_mode.shape[1]):
                # slices 1 thru 155
                slices_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix]
        return slices_by_mode, slices_by_slice

    def norm_slices(self):
        '''
        normalizes each slice in self.slices_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
        if n4itk == True, will apply n4itk bias correction to T1 and T1c images
        '''
        print 'Normalizing slices...'
        normed_slices = np.zeros((155, 5, 240, 240))
        for slice_ix in xrange(155):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in xrange(4):
                normed_slices[slice_ix][mode_ix] = self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print 'Done.'
        return normed_slices

    def _normalize(self, slice):
        '''
        INPUT:  (1) a single slice of any given modality (excluding gt)
                (2) index of modality assoc with slice (0=flair, 1=t1, 2=t1c, 3=t2)
        OUTPUT: normalized slice
        '''
        # b, t = np.percentile(slice, (0.5, 99.5))
        # slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)

    def save_patient(self, reg_norm_n4, patient_num, modality,
                     dst_dir,
                     clip_idx, crop_bd,
                     save_mode='all'):

        print 'Saving scans fo` patient {}...'.format(patient_num)

        dst_H5_path = os.path.join(dst_dir, 'gt', reg_norm_n4 + '_JPG')
        if not os.path.exists(dst_H5_path):
            os.makedirs(dst_H5_path)

        if save_mode == 'all':
            dst_0_path = os.path.join(dst_dir, 'flair', reg_norm_n4 + '_JPG')
            dst_1_path = os.path.join(dst_dir, 't1', reg_norm_n4 + '_JPG')
            dst_2_path = os.path.join(dst_dir, 't1s', reg_norm_n4 + '_JPG')
            dst_3_path = os.path.join(dst_dir, 't2', reg_norm_n4 + '_JPG')
            if not os.path.exists(dst_0_path):
                os.makedirs(dst_0_path)
            if not os.path.exists(dst_1_path):
                os.makedirs(dst_1_path)
            if not os.path.exists(dst_2_path):
                os.makedirs(dst_2_path)
            if not os.path.exists(dst_3_path):
                os.makedirs(dst_3_path)

            dst_dir_path = [dst_0_path, dst_1_path, dst_2_path, dst_3_path]

        else:
            modality_idx = config.MODALITY_DICT[modality]
            dst_dir_path = os.path.join(dst_dir, modality, reg_norm_n4 + '_JPG')
            if not os.path.exists(dst_dir_path):
                os.makedirs(dst_dir_path)

        if reg_norm_n4 == 'reg':
            for slice_ix in xrange(155):
                gt_strip = self.slices_by_slice[slice_ix][4]
                if np.max(gt_strip) != 0:
                    # +++++++++++++++++++++ STORE all modes ++++++++++++++++
                    if save_mode == 'all':
                        for s in xrange(4):
                            strip = self.slices_by_slice[slice_ix][s]
                            strip /= np.max(strip)
                            strip = strip[crop_bd:224 - crop_bd, crop_bd:224 - crop_bd]
                            io.imsave(dst_dir_path[s] + '/{}_{}.jpg'.format(patient_num, slice_ix),
                                      strip)
                    else:
                        strip = self.slices_by_slice[slice_ix][modality_idx]
                        strip /= np.max(strip)
                        strip = strip[crop_bd:224 - crop_bd, crop_bd:224 - crop_bd]
                        io.imsave(dst_dir_path +
                                  '/{}_{}.jpg'.format(patient_num, slice_ix), strip)

                    # +++++++++++++++++++++ STORE gt +++++++++++++++++++++++
                    gt_strip = gt_strip[crop_bd:224 - crop_bd, crop_bd:224 - crop_bd]
                    H5_name = dst_H5_path +\
                        '/{}_{}.h5'.format(patient_num, slice_ix)
                    h5f = h5py.File(H5_name, 'w')
                    h5f.create_dataset('gt', data=gt_strip)
                    h5f.close()

        elif reg_norm_n4 == 'norm':
            for slice_ix in xrange(155):
                strip = self.normed_slices[slice_ix][modality_idx]
                gt_strip = self.normed_slices[slice_ix][4]
                if np.max(gt_strip) != 0:  # set values < 1
                    strip /= np.max(strip)

                # ========================================================
                # checl this step
                if np.min(strip) <= -1:  # set values > -1
                    strip /= abs(np.min(strip))
                # ========================================================
                # save as patient_slice.png
                strip = strip[crop_bd:224 - crop_bd, crop_bd:224 - crop_bd]
                io.imsave(dst_dir_path + '/{}_{}.jpg'.format(patient_num, slice_ix),
                          strip)

        else:
            for slice_ix in xrange(155):
                strip = self.normed_slices[slice_ix][modality_idx]
                gt_strip = self.normed_slices[slice_ix][4]
                # if np.max(strip) != 0:  # set values < 1
                #     strip /= np.max(strip)
                # if np.min(strip) <= -1:  # set values > -1
                if np.max(gt_strip) != 0:
                    strip /= abs(np.min(strip))
                    strip = strip[crop_bd:224 - crop_bd, crop_bd:224 - crop_bd]
                    io.imsave(dst_dir_path + '/{}_{}.jpg'.format(patient_num, slice_ix),
                              strip)


def save_patient_slices(patients, img_mode, modality,
                        dst_dir,
                        clip_idx, crop_bd,
                        save_mode):

    for patient_num, path in enumerate(patients):
        a = BrainPipeline(path)
        a.save_patient(img_mode, patient_num, modality,
                       dst_dir,
                       clip_idx, crop_bd,
                       save_mode)


if __name__ == '__main__':

    PER = 7
    CROP_BUNDRY = 20
    # 184x184

    print config.MODALITY_DICT[config.MODALITY]

    save_patient_slices(glob(config.SRC_NIFTY_DIR)[:2],
                        config.IMG_MODE,    # -> whether 'norm' or 'reg'
                        config.MODALITY,
                        config.DST_JPG_DIR,  # -> Root dir for saving
                        PER, CROP_BUNDRY,   # -> dst image specs
                        save_mode='all')
    # save_mode='single')  -> 't1' or 't2'
