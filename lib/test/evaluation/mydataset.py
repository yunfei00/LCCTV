import numpy as np

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class myDataset(BaseDataset):
    """
    OTB-2015 dataset.

    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf

    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.my_path
        # self.video_path = self.env_settings.video_path
        # self.video_name = self.env_settings.video_name
        # self.video_len = self.env_settings.video_len
        self.sequence_info_list = self._get_sequence_info_list()
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])
    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']
        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']
        # frames = ['{sequence_path}/{frame:0{nz}}.{ext}'.format(sequence_path=sequence_path, frame=frame_num,
        #                                                         nz=nz, ext=ext) for frame_num in
        #                                                         range(start_frame + init_omit, end_frame + 1)]
        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]
        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # anno_path = '{}'.format(sequence_info['anno_path'])

        # Note: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        # return Sequence(sequence_info['name'], frames, 'my', ground_truth_rect[init_omit:, :],
        #                 object_class=sequence_info['object_class'])

        # 只含第一帧的gt
        return Sequence(sequence_info['name'], frames, 'my', ground_truth_rect[np.newaxis, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'real1', 'path': '1/imgs', 'startFrame': 1, 'endFrame': 100, 'nz': 8,
             'ext': 'jpg',
             'anno_path': '1/groundtruth.txt',
             'object_class': 'real1'},
            {'name': 'real1', 'path': '2_all/img', 'startFrame': 1, 'endFrame': 100, 'nz': 8,
             'ext': 'jpg',
             'anno_path': '2_all/groundtruth.txt',
             'object_class': 'real1'},
            {'name': 'real1', 'path': '3/img', 'startFrame': 1, 'endFrame': 100, 'nz': 8,
             'ext': 'jpg',
             'anno_path': '3/groundtruth.txt',
             'object_class': 'real1'},
        ]
        return sequence_info_list